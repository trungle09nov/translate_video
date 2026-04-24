// ─── Tab Navigation ───────────────────────────────────────────────────────────
document.querySelectorAll('.nav-item').forEach(btn => {
  btn.addEventListener('click', () => {
    const tabId = btn.dataset.tab;
    document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(`tab-${tabId}`).classList.add('active');
  });
});

// ─── File Drop Helper ─────────────────────────────────────────────────────────
function setupFileDrop(dropId, inputId, badgeId) {
  const dropEl = document.getElementById(dropId);
  const inputEl = document.getElementById(inputId);
  const badgeEl = document.getElementById(badgeId);

  dropEl.addEventListener('click', () => inputEl.click());

  dropEl.addEventListener('dragover', e => {
    e.preventDefault();
    dropEl.classList.add('drag-over');
  });

  dropEl.addEventListener('dragleave', () => dropEl.classList.remove('drag-over'));

  dropEl.addEventListener('drop', e => {
    e.preventDefault();
    dropEl.classList.remove('drag-over');
    if (e.dataTransfer.files[0]) {
      inputEl.files = e.dataTransfer.files;
      showBadge(badgeEl, e.dataTransfer.files[0].name);
    }
  });

  inputEl.addEventListener('change', () => {
    if (inputEl.files[0]) showBadge(badgeEl, inputEl.files[0].name);
  });
}

function showBadge(badgeEl, name) {
  badgeEl.textContent = '📎 ' + name;
  badgeEl.classList.remove('hidden');
}

setupFileDrop('video-drop', 'video-file', 'video-filename');
setupFileDrop('transcript-drop', 'transcript-file', 'transcript-filename');
setupFileDrop('transcript-audio-drop', 'transcript-audio-file', 'transcript-audio-filename');
setupFileDrop('dubbing-drop', 'dubbing-file', 'dubbing-filename');

// ─── Transcript Mode Toggle ───────────────────────────────────────────────────
function toggleTranscriptMode(radio) {
  const isAudio = radio.value === 'audio';
  document.getElementById('transcript-file-section').classList.toggle('hidden', isAudio);
  document.getElementById('transcript-audio-section').classList.toggle('hidden', !isAudio);
}

// ─── API Key Toggle ───────────────────────────────────────────────────────────
function toggleKeyVisibility() {
  const input = document.getElementById('elevenlabs-key');
  input.type = input.type === 'password' ? 'text' : 'password';
}

// ─── Log / Progress Helpers ───────────────────────────────────────────────────
function appendLog(logId, message, type = '') {
  const logBox = document.getElementById(logId);
  const placeholder = logBox.querySelector('.log-placeholder');
  if (placeholder) placeholder.remove();

  const line = document.createElement('div');
  line.className = `log-line ${type}`;
  line.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
  logBox.appendChild(line);
  logBox.scrollTop = logBox.scrollHeight;
}

function setProgress(barId, labelId, progress, status = 'running') {
  const bar = document.getElementById(barId);
  const label = document.getElementById(labelId);
  bar.style.width = progress + '%';
  bar.className = 'progress-bar';
  if (status === 'done') bar.classList.add('done');
  if (status === 'error') bar.classList.add('error');
  label.textContent = `${progress}%`;
}

// ─── SSE Job Listener ─────────────────────────────────────────────────────────
function listenJob(jobId, { barId, labelId, logId, onDone, onError }) {
  const es = new EventSource(`/api/stream/${jobId}`);

  es.onmessage = e => {
    const data = JSON.parse(e.data);
    const { status, progress, logs, result } = data;

    setProgress(barId, labelId, progress, status);

    (logs || []).forEach(msg => {
      const type = msg.toLowerCase().includes('error') ? 'error'
                 : status === 'done' && msg === 'Done!' ? 'done' : '';
      appendLog(logId, msg, type);
    });

    if (status === 'done') {
      es.close();
      appendLog(logId, 'Hoàn thành!', 'done');
      if (onDone) onDone(result);
    } else if (status === 'error') {
      es.close();
      appendLog(logId, 'Đã xảy ra lỗi.', 'error');
      if (onError) onError();
    }
  };

  es.onerror = () => {
    es.close();
    appendLog(logId, 'Mất kết nối SSE', 'error');
  };
}

// ─── TAB 1: Video Translation ─────────────────────────────────────────────────
document.getElementById('video-form').addEventListener('submit', async e => {
  e.preventDefault();
  const fileInput = document.getElementById('video-file');
  if (!fileInput.files[0]) {
    alert('Vui lòng chọn file video!');
    return;
  }

  const form = e.target;
  const fd = new FormData();
  fd.append('file', fileInput.files[0]);
  fd.append('src_lang', document.getElementById('v-src-lang').value);
  fd.append('tgt_lang', document.getElementById('v-tgt-lang').value);
  fd.append('fps', form.querySelector('[name="fps"]').value);
  fd.append('num_gpus', form.querySelector('[name="num_gpus"]').value);
  fd.append('font_path', form.querySelector('[name="font_path"]').value);

  form.querySelector('.btn-primary').disabled = true;
  document.getElementById('video-result').classList.add('hidden');
  document.getElementById('video-log').innerHTML = '';

  const res = await fetch('/api/video/translate', { method: 'POST', body: fd });
  const { job_id } = await res.json();

  appendLog('video-log', `Job started: ${job_id}`);
  setProgress('video-progress-bar', 'video-progress-label', 0);

  listenJob(job_id, {
    barId: 'video-progress-bar',
    labelId: 'video-progress-label',
    logId: 'video-log',
    onDone: result => {
      form.querySelector('.btn-primary').disabled = false;
      if (result?.video) {
        document.getElementById('video-download-link').href = result.video;
        document.getElementById('video-result').classList.remove('hidden');
      }
    },
    onError: () => { form.querySelector('.btn-primary').disabled = false; },
  });
});

// ─── TAB 2: Transcript Translation ───────────────────────────────────────────
document.getElementById('transcript-form').addEventListener('submit', async e => {
  e.preventDefault();
  const form = e.target;
  const isAudio = form.querySelector('[name="mode"]:checked').value === 'audio';

  const fd = new FormData();
  fd.append('src_lang', document.getElementById('t-src-lang').value);
  fd.append('tgt_lang', document.getElementById('t-tgt-lang').value);
  fd.append('output_format', form.querySelector('[name="out_format"]:checked').value);
  fd.append('use_whisper', isAudio ? 'true' : 'false');

  if (isAudio) {
    const audioFile = document.getElementById('transcript-audio-file').files[0];
    if (!audioFile) { alert('Vui lòng chọn file audio!'); return; }
    fd.append('audio_file', audioFile);
  } else {
    const txtFile = document.getElementById('transcript-file').files[0];
    if (!txtFile) { alert('Vui lòng chọn file transcript!'); return; }
    fd.append('file', txtFile);
  }

  form.querySelector('.btn-primary').disabled = true;
  document.getElementById('transcript-result').classList.add('hidden');
  document.getElementById('transcript-log').innerHTML = '';

  const res = await fetch('/api/transcript/translate', { method: 'POST', body: fd });
  const { job_id } = await res.json();

  appendLog('transcript-log', `Job started: ${job_id}`);
  setProgress('transcript-progress-bar', 'transcript-progress-label', 0);

  listenJob(job_id, {
    barId: 'transcript-progress-bar',
    labelId: 'transcript-progress-label',
    logId: 'transcript-log',
    onDone: result => {
      form.querySelector('.btn-primary').disabled = false;
      if (result?.file) {
        document.getElementById('transcript-download-link').href = result.file;
        if (result.preview) {
          document.getElementById('transcript-preview').textContent = result.preview + (result.preview.length >= 500 ? '...' : '');
        }
        document.getElementById('transcript-result').classList.remove('hidden');
      }
    },
    onError: () => { form.querySelector('.btn-primary').disabled = false; },
  });
});

// ─── TAB 3: Audio Dubbing ─────────────────────────────────────────────────────
document.getElementById('dubbing-form').addEventListener('submit', async e => {
  e.preventDefault();
  const form = e.target;
  const fileInput = document.getElementById('dubbing-file');
  const apiKey = document.getElementById('elevenlabs-key').value.trim();

  if (!fileInput.files[0]) { alert('Vui lòng chọn file audio/video!'); return; }
  if (!apiKey) { alert('Vui lòng nhập ElevenLabs API Key!'); return; }

  const fd = new FormData();
  fd.append('file', fileInput.files[0]);
  fd.append('api_key', apiKey);
  fd.append('src_lang', document.getElementById('d-src-lang').value);
  fd.append('tgt_lang', document.getElementById('d-tgt-lang').value);
  fd.append('start_time', document.getElementById('start-time').value || '0');
  fd.append('end_time', document.getElementById('end-time').value || '0');
  fd.append('num_speakers', document.getElementById('num-speakers').value || '0');

  form.querySelector('.btn-primary').disabled = true;
  document.getElementById('dubbing-result').classList.add('hidden');
  document.getElementById('dubbing-log').innerHTML = '';

  const res = await fetch('/api/audio/dub', { method: 'POST', body: fd });
  const { job_id } = await res.json();

  appendLog('dubbing-log', `Job started: ${job_id}`);
  setProgress('dubbing-progress-bar', 'dubbing-progress-label', 0);

  listenJob(job_id, {
    barId: 'dubbing-progress-bar',
    labelId: 'dubbing-progress-label',
    logId: 'dubbing-log',
    onDone: result => {
      form.querySelector('.btn-primary').disabled = false;
      if (result?.audio) {
        const player = document.getElementById('dubbing-audio-player');
        player.src = result.audio;
        document.getElementById('dubbing-download-link').href = result.audio;
        document.getElementById('dubbing-result').classList.remove('hidden');
      }
    },
    onError: () => { form.querySelector('.btn-primary').disabled = false; },
  });
});
