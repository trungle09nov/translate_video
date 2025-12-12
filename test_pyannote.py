# test_pyannote_real_audio_fixed.py
import sys
import os
import torch
from dotenv import load_dotenv

load_dotenv()
# FIX FOR PYTORCH 2.6+
print("Applying PyTorch 2.6+ fix...")
if hasattr(torch, 'serialization'):
    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    from collections import OrderedDict
    torch.serialization.add_safe_globals([OrderedDict])
print("✓ Fix applied\n")

def test_diarization(audio_path, hf_token=None, num_speakers=None):
    """Test diarization with real audio - PyTorch 2.6+ compatible"""
    
    print("="*80)
    print("PYANNOTE DIARIZATION TEST (PyTorch 2.6+ Compatible)")
    print("="*80)
    
    # Check audio file
    if not os.path.exists(audio_path):
        print(f"❌ Audio file not found: {audio_path}")
        return False
    
    print(f"\n✓ Audio file: {audio_path}")
    
    # Get token
    if not hf_token:
        hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
    
    if not hf_token:
        print("❌ No HF token! Set with: export HF_TOKEN=your_token")
        return False
    
    print(f"✓ HF Token: {hf_token[:10]}...")
    
    # Load pipeline
    print("\n[1/3] Loading pipeline...")
    try:
        from pyannote.audio import Pipeline
        
        # Monkey patch torch.load to use weights_only=False for pyannote
        original_load = torch.load
        
        def patched_load(*args, **kwargs):
            # Force weights_only=False for pyannote models
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        
        torch.load = patched_load
        
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        
        # Restore original torch.load
        torch.load = original_load
        
        # Move to GPU if available
        if torch.cuda.is_available():
            device = torch.device("cuda:1")
            pipeline.to(device)
            print(f"✅ Pipeline loaded on {device}")
        else:
            print("✅ Pipeline loaded on CPU")
            
    except Exception as e:
        print(f"❌ Failed to load pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Run diarization
    print("\n[2/3] Running diarization...")
    print("   (This may take a few minutes for long audio...)")
    try:
        diarization_kwargs = {}
        if num_speakers:
            diarization_kwargs['num_speakers'] = num_speakers
        
        diarization = pipeline(audio_path, **diarization_kwargs)
        print("✅ Diarization complete!")
        
    except Exception as e:
        print(f"❌ Diarization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Analyze results
    print("\n[3/3] Analyzing results...")
    try:
        segments = []
        speakers = set()
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                'start': turn.start,
                'end': turn.end,
                'speaker': speaker
            })
            speakers.add(speaker)
        
        print(f"\n✅ Found {len(speakers)} speakers: {sorted(speakers)}")
        print(f"✅ Total segments: {len(segments)}")
        
        # Show first few segments
        print("\n📝 First 10 segments:")
        for i, seg in enumerate(segments[:10], 1):
            duration = seg['end'] - seg['start']
            print(f"  {i:2d}. [{seg['start']:6.2f}s - {seg['end']:6.2f}s] ({duration:5.2f}s) {seg['speaker']}")
        
        if len(segments) > 10:
            print(f"  ... and {len(segments) - 10} more segments")
        
        # Speaker statistics
        print("\n📊 Speaker statistics:")
        speaker_stats = {}
        for seg in segments:
            speaker = seg['speaker']
            duration = seg['end'] - seg['start']
            
            if speaker not in speaker_stats:
                speaker_stats[speaker] = {'time': 0, 'segments': 0}
            
            speaker_stats[speaker]['time'] += duration
            speaker_stats[speaker]['segments'] += 1
        
        for speaker in sorted(speaker_stats.keys()):
            stats = speaker_stats[speaker]
            print(f"  {speaker}:")
            print(f"    - Total time: {stats['time']:.2f}s ({stats['time']/60:.2f}min)")
            print(f"    - Segments: {stats['segments']}")
            print(f"    - Avg segment: {stats['time']/stats['segments']:.2f}s")
        
        # Save results
        import json
        output_file = audio_path.rsplit('.', 1)[0] + '_diarization.json'
        with open(output_file, 'w') as f:
            json.dump({
                'speakers': sorted(speakers),
                'segments': segments,
                'statistics': speaker_stats
            }, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
        
        print("\n" + "="*80)
        print("✅ TEST SUCCESSFUL!")
        print("="*80)
        
        return segments
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_pyannote.py <audio_file> [hf_token] [num_speakers]")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    hf_token = sys.argv[2] if len(sys.argv) > 2 else None
    num_speakers = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    if num_speakers:
        print(f"ℹ️  Forcing {num_speakers} speakers\n")
    
    success = test_diarization(audio_path, hf_token, num_speakers)
    
    if not success:
        sys.exit(1)