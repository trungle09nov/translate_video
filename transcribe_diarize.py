# whisper_diarization_pipeline.py
import whisper
import torch
from pathlib import Path
import json
import csv
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PYTORCH 2.6+ FIX - Enhanced
# ============================================================================
print("Applying PyTorch 2.6+ compatibility fix...")

if hasattr(torch, 'serialization'):
    # Add basic torch types
    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    from collections import OrderedDict
    torch.serialization.add_safe_globals([OrderedDict])
    
    # Add pyannote-specific types
    try:
        from pyannote.audio.core.task import Specifications
        torch.serialization.add_safe_globals([Specifications])
        print("  ✓ Added Specifications")
    except:
        pass
    
    try:
        from pyannote.core import Segment, Timeline, Annotation
        torch.serialization.add_safe_globals([Segment, Timeline, Annotation])
        print("  ✓ Added pyannote.core types")
    except:
        pass

# Monkey patch torch.load to use weights_only=False
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

print("✓ PyTorch 2.6+ fix applied\n")

from pyannote.audio import Pipeline


class WhisperDiarizationPipeline:
    """
    Complete pipeline: Audio → Transcription + Speaker Diarization
    """
    
    def __init__(
        self,
        whisper_model="large-v2",
        device="cuda:1",
        hf_token=None
    ):
        """
        Initialize pipeline
        
        Args:
            whisper_model: Whisper model (tiny, base, small, medium, large, large-v2, large-v3)
            device: GPU device (cuda:0, cuda:1, etc)
            hf_token: HuggingFace token for speaker diarization
        """
        self.device = device
        self.hf_token = hf_token
        
        # Load Whisper
        print(f"🎯 Loading Whisper '{whisper_model}' on {device}...")
        self.whisper_model = whisper.load_model(whisper_model, device=device)
        print("✅ Whisper loaded")
        
        # Load Pyannote Diarization
        self.diarization_pipeline = None
        if hf_token:
            try:
                print("🎤 Loading speaker diarization pipeline...")
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=hf_token
                )
                self.diarization_pipeline.to(torch.device(device))
                print("✅ Diarization loaded")
            except Exception as e:
                print(f"⚠️  Diarization failed to load: {e}")
                print("    Will continue without speaker labels")
                import traceback
                traceback.print_exc()
        else:
            print("⚠️  No HF token - speakers will not be identified")
    
    def process(
        self,
        audio_path,
        language="de",
        min_speakers=None,
        max_speakers=None,
        output_dir="output"
    ):
        """
        Process audio: transcribe + identify speakers
        
        Args:
            audio_path: Path to audio file
            language: Language code (de, en, vi, etc)
            min_speakers: Minimum number of speakers (None = auto)
            max_speakers: Maximum number of speakers (None = auto)
            output_dir: Directory to save results
            
        Returns:
            Dict with transcription and speaker info
        """
        audio_path = Path(audio_path)
        
        print("\n" + "="*80)
        print(f"PROCESSING: {audio_path.name}")
        print("="*80)
        
        # Step 1: Transcribe
        print("\n[1/3] 📝 Transcribing audio...")
        transcription = self._transcribe(str(audio_path), language)
        
        # Step 2: Diarize (identify speakers)
        print("\n[2/3] 🎤 Identifying speakers...")
        diarization = self._diarize(str(audio_path), min_speakers, max_speakers)
        
        # Step 3: Align speakers to transcription
        print("\n[3/3] 🔗 Aligning speakers to text...")
        result = self._align_speakers(transcription, diarization)
        
        # Save results
        print("\n💾 Saving results...")
        output_files = self._save_results(result, output_dir, audio_path.stem)
        
        # Print summary
        self._print_summary(result)
        
        print("\n" + "="*80)
        print("✅ PROCESSING COMPLETE!")
        print("="*80)
        print(f"\nOutput files saved to: {output_dir}/")
        for key, path in output_files.items():
            print(f"  - {key}: {path.name}")
        
        return result
    
    def _transcribe(self, audio_path, language):
        """Transcribe audio with Whisper"""
        result = self.whisper_model.transcribe(
            audio_path,
            language=language,
            verbose=False,
            word_timestamps=True,
            temperature=0.0
        )
        
        print(f"   ✓ Transcribed {len(result['segments'])} segments")
        print(f"   ✓ Detected language: {result.get('language', 'unknown')}")
        
        return result
    
    def _diarize(self, audio_path, min_speakers=None, max_speakers=None):
        """Identify speakers in audio"""
        if not self.diarization_pipeline:
            print("   ⚠️  Diarization not available (no speakers identified)")
            return None
        
        try:
            # Run diarization
            diarization_params = {}
            if min_speakers:
                diarization_params['min_speakers'] = min_speakers
            if max_speakers:
                diarization_params['max_speakers'] = max_speakers
            
            diarization = self.diarization_pipeline(audio_path, **diarization_params)
            
            # Convert to list of segments
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    'start': turn.start,
                    'end': turn.end,
                    'speaker': speaker
                })
            
            speakers = set(seg['speaker'] for seg in segments)
            print(f"   ✓ Detected {len(speakers)} speakers: {sorted(speakers)}")
            print(f"   ✓ Total speaker segments: {len(segments)}")
            
            return segments
            
        except Exception as e:
            print(f"   ⚠️  Diarization failed: {e}")
            return None
    
    def _align_speakers(self, transcription, diarization):
        """Align speaker labels to transcription segments"""
        if not diarization:
            for segment in transcription['segments']:
                segment['speaker'] = 'SPEAKER'
            return transcription
        
        for segment in transcription['segments']:
            segment_mid = (segment['start'] + segment['end']) / 2
            
            best_speaker = None
            best_overlap = 0
            
            for diar_seg in diarization:
                overlap_start = max(segment['start'], diar_seg['start'])
                overlap_end = min(segment['end'], diar_seg['end'])
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = diar_seg['speaker']
            
            segment['speaker'] = best_speaker if best_speaker else 'UNKNOWN'
        
        if 'words' in segment and segment.get('words'):
            for word in segment['words']:
                word_start = word.get('start', segment['start'])
                word_end = word.get('end', segment['end'])
                word_mid = (word_start + word_end) / 2
                
                word_speaker = segment['speaker']
                for diar_seg in diarization:
                    if diar_seg['start'] <= word_mid <= diar_seg['end']:
                        word_speaker = diar_seg['speaker']
                        break
                
                word['speaker'] = word_speaker
        
        speakers = set(seg.get('speaker') for seg in transcription['segments'])
        print(f"   ✓ Aligned {len(speakers)} speakers to transcription")
        
        return transcription
    
    def _print_summary(self, result):
        """Print summary of results"""
        speaker_stats = {}
        for segment in result['segments']:
            speaker = segment.get('speaker', 'UNKNOWN')
            
            if speaker not in speaker_stats:
                speaker_stats[speaker] = {
                    'segments': 0,
                    'duration': 0,
                    'words': 0,
                    'sample_text': []
                }
            
            stats = speaker_stats[speaker]
            stats['segments'] += 1
            stats['duration'] += segment['end'] - segment['start']
            
            if 'words' in segment:
                stats['words'] += len(segment['words'])
            
            if len(stats['sample_text']) < 2:
                stats['sample_text'].append(segment['text'].strip())
        
        print("\n📊 SUMMARY")
        print("-" * 80)
        
        total_duration = result['segments'][-1]['end'] if result['segments'] else 0
        print(f"Total duration: {total_duration:.2f}s ({total_duration/60:.2f}min)")
        print(f"Total segments: {len(result['segments'])}")
        print(f"Speakers: {len(speaker_stats)}")
        
        print("\n👥 SPEAKER BREAKDOWN:")
        for speaker in sorted(speaker_stats.keys()):
            stats = speaker_stats[speaker]
            print(f"\n  {speaker}:")
            print(f"    - Segments: {stats['segments']}")
            print(f"    - Duration: {stats['duration']:.2f}s ({stats['duration']/60:.2f}min)")
            print(f"    - % of total: {stats['duration']/total_duration*100:.1f}%")
            if stats['words'] > 0:
                print(f"    - Words: {stats['words']}")
            if stats['sample_text']:
                print(f"    - Sample: \"{stats['sample_text'][0][:80]}...\"")
    
    def _save_results(self, result, output_dir, audio_name):
        """Save results in multiple formats"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_files = {}
        
        # 1. Full JSON
        json_path = output_dir / f"{audio_name}_full.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        output_files['json'] = json_path
        
        # 2. Transcript
        txt_path = output_dir / f"{audio_name}_transcript.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            current_speaker = None
            for segment in result['segments']:
                speaker = segment.get('speaker', 'UNKNOWN')
                text = segment['text'].strip()
                
                if speaker != current_speaker:
                    if current_speaker is not None:
                        f.write("\n")
                    current_speaker = speaker
                
                f.write(f"{speaker} [{segment['start']:.2f}s - {segment['end']:.2f}s]: {text}\n")
        output_files['transcript'] = txt_path
        
        # 3. Simple text
        simple_path = output_dir / f"{audio_name}_text.txt"
        with open(simple_path, 'w', encoding='utf-8') as f:
            for segment in result['segments']:
                f.write(segment['text'].strip() + "\n")
        output_files['text'] = simple_path
        
        # 4. CSV
        csv_path = output_dir / f"{audio_name}_segments.csv"
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['start', 'end', 'duration', 'speaker', 'text'])
            
            for segment in result['segments']:
                writer.writerow([
                    f"{segment['start']:.3f}",
                    f"{segment['end']:.3f}",
                    f"{segment['end'] - segment['start']:.3f}",
                    segment.get('speaker', 'UNKNOWN'),
                    segment['text'].strip()
                ])
        output_files['csv'] = csv_path
        
        # 5. SRT subtitles
        srt_path = output_dir / f"{audio_name}_subtitles.srt"
        with open(srt_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(result['segments'], 1):
                speaker = segment.get('speaker', 'UNKNOWN')
                text = segment['text'].strip()
                
                start = self._format_srt_time(segment['start'])
                end = self._format_srt_time(segment['end'])
                
                f.write(f"{i}\n")
                f.write(f"{start} --> {end}\n")
                f.write(f"[{speaker}] {text}\n")
                f.write("\n")
        output_files['srt'] = srt_path
        
        return output_files
    
    def _format_srt_time(self, seconds):
        """Format seconds to SRT timestamp format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Whisper + Diarization Pipeline: Audio → Text + Speakers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python whisper_diarization_pipeline.py audio.wav --hf-token YOUR_TOKEN
  
  # Specify 2 speakers
  python whisper_diarization_pipeline.py audio.wav --hf-token YOUR_TOKEN --max-speakers 2
  
  # Use medium model (faster)
  python whisper_diarization_pipeline.py audio.wav --hf-token YOUR_TOKEN --model medium
        """
    )
    
    parser.add_argument('audio_path', help='Path to audio file')
    parser.add_argument('--hf-token', required=True, help='HuggingFace token')
    parser.add_argument('--model', default='large-v2', 
                       choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'],
                       help='Whisper model (default: large-v2)')
    parser.add_argument('--language', default='de', help='Language code (default: de)')
    parser.add_argument('--device', default='cuda:2', help='GPU device')
    parser.add_argument('--min-speakers', type=int, help='Minimum speakers')
    parser.add_argument('--max-speakers', type=int, help='Maximum speakers')
    parser.add_argument('--output-dir', default='output', help='Output directory')
    
    args = parser.parse_args()
    
    pipeline = WhisperDiarizationPipeline(
        whisper_model=args.model,
        device=args.device,
        hf_token=args.hf_token
    )
    
    result = pipeline.process(
        audio_path=args.audio_path,
        language=args.language,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()