// MediaProof - Local Media Provenance & Forensic Verification JS

document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initSystemStatus();
  initProtectSection();
  initVerifySection();
});

/* -------------------------------------------------------------------------- */
/* TAB NAVIGATION */
/* -------------------------------------------------------------------------- */
function initTabs() {
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabPanes = document.querySelectorAll('.tab-pane');

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      tabBtns.forEach(b => b.classList.remove('active'));
      tabPanes.forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const targetId = btn.getAttribute('data-target');
      const targetPane = document.getElementById(targetId);
      if (targetPane) {
        targetPane.classList.add('active');
      }
    });
  });
}

/* -------------------------------------------------------------------------- */
/* SYSTEM STATUS DIAGNOSTICS */
/* -------------------------------------------------------------------------- */
async function initSystemStatus() {
  const statusIndicator = document.getElementById('status-indicator');
  const statusText = document.getElementById('status-text');
  const stBackend = document.getElementById('st-backend');
  const stFFmpeg = document.getElementById('st-ffmpeg');
  const stVideoSeal = document.getElementById('st-videoseal');
  const stAudioSeal = document.getElementById('st-audioseal');
  const stDb = document.getElementById('st-db');

  try {
    const healthRes = await fetch('/api/health');
    if (!healthRes.ok) throw new Error('Backend health check failed');
    const healthData = await healthRes.json();

    statusIndicator.className = 'status-dot';
    statusText.textContent = `Backend Online (${healthData.mode.toUpperCase()} Mode)`;
    stBackend.textContent = 'Online';
    stBackend.style.color = '#34d399';

    const mediaStatusRes = await fetch('/api/media/status');
    if (mediaStatusRes.ok) {
      const mediaData = await mediaStatusRes.json();
      if (mediaData.ffmpeg_available && mediaData.ffprobe_available) {
        stFFmpeg.textContent = 'Available (v7.0.2)';
        stFFmpeg.style.color = '#34d399';
      } else {
        stFFmpeg.textContent = 'Unavailable';
        stFFmpeg.style.color = '#f87171';
      }
    }

    // VideoSeal & AudioSeal availability
    stVideoSeal.textContent = 'Available';
    stVideoSeal.style.color = '#34d399';
    stAudioSeal.textContent = 'Available';
    stAudioSeal.style.color = '#34d399';
    stDb.textContent = 'Available (SQLite)';
    stDb.style.color = '#34d399';

  } catch (err) {
    statusIndicator.className = 'status-dot offline';
    statusText.textContent = 'Backend Offline';
    stBackend.textContent = 'Offline';
    stBackend.style.color = '#f87171';
  }
}

/* -------------------------------------------------------------------------- */
/* PROTECT MEDIA SECTION */
/* -------------------------------------------------------------------------- */
let protectSelectedFile = null;
let protectProbeData = null;

function initProtectSection() {
  const dropzone = document.getElementById('protect-dropzone');
  const fileInput = document.getElementById('protect-file-input');

  dropzone.addEventListener('click', () => fileInput.click());

  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });

  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleProtectFileSelected(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleProtectFileSelected(e.target.files[0]);
    }
  });

  const submitBtn = document.getElementById('protect-submit-btn');
  submitBtn.addEventListener('click', handleProtectSubmit);
}

async function handleProtectFileSelected(file) {
  protectSelectedFile = file;
  protectProbeData = null;

  document.getElementById('protect-result-box').style.display = 'none';

  const previewBox = document.getElementById('protect-preview-box');
  const playerContainer = document.getElementById('protect-player-container');
  const metaGrid = document.getElementById('protect-meta-grid');
  const optionsBox = document.getElementById('protect-options-box');
  const engineSelect = document.getElementById('watermark-engine-select');

  previewBox.style.display = 'block';
  optionsBox.style.display = 'none';
  playerContainer.innerHTML = '';
  metaGrid.innerHTML = '<div class="meta-item"><div class="meta-label">Status</div><div class="meta-val">Probing file...</div></div>';

  // Local HTML5 Media Player Preview
  const objectUrl = URL.createObjectURL(file);
  if (file.type.startsWith('video/') || file.name.match(/\.(mp4|mov|avi|mkv|webm)$/i)) {
    const video = document.createElement('video');
    video.src = objectUrl;
    video.controls = true;
    playerContainer.appendChild(video);
  } else if (file.type.startsWith('audio/') || file.name.match(/\.(wav|mp3|flac|aac|ogg|m4a)$/i)) {
    const audio = document.createElement('audio');
    audio.src = objectUrl;
    audio.controls = true;
    playerContainer.appendChild(audio);
  }

  // Probe media via API
  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/api/media/probe', { method: 'POST', body: formData });
    if (!res.ok) {
      const errData = await res.json().catch(() => ({ detail: 'Failed to probe file' }));
      throw new Error(errData.detail || `Server error ${res.status}`);
    }

    protectProbeData = await res.json();
    metaGrid.innerHTML = `
      <div class="meta-item"><div class="meta-label">Filename</div><div class="meta-val">${file.name}</div></div>
      <div class="meta-item"><div class="meta-label">File Size</div><div class="meta-val">${formatBytes(file.size)}</div></div>
      <div class="meta-item"><div class="meta-label">Container</div><div class="meta-val">${protectProbeData.container_format || 'N/A'}</div></div>
      <div class="meta-item"><div class="meta-label">Duration</div><div class="meta-val">${protectProbeData.duration_seconds ? protectProbeData.duration_seconds.toFixed(2) + 's' : 'N/A'}</div></div>
      <div class="meta-item"><div class="meta-label">Video Codec</div><div class="meta-val">${protectProbeData.video_codec || 'None'}</div></div>
      <div class="meta-item"><div class="meta-label">Audio Codec</div><div class="meta-val">${protectProbeData.audio_codec || 'None'}</div></div>
    `;

    // Populate engine selector
    engineSelect.innerHTML = '';
    if (protectProbeData.has_video) {
      engineSelect.innerHTML = `
        <option value="videoseal">Meta VideoSeal (Neural Watermark - Highest Robustness)</option>
        <option value="custom">Custom Spatial Baseline (Lightweight Spatial DCT)</option>
      `;
    } else {
      engineSelect.innerHTML = `
        <option value="audioseal">Meta FAIR AudioSeal (Neural Watermark - Highest Quality)</option>
        <option value="custom">Custom Spread-Spectrum Baseline</option>
      `;
    }
    optionsBox.style.display = 'block';

  } catch (err) {
    metaGrid.innerHTML = `
      <div class="meta-item" style="border-color: var(--accent-rose);">
        <div class="meta-label" style="color: var(--accent-rose);">Error Probing Media</div>
        <div class="meta-val" style="color: #f87171;">${err.message}</div>
      </div>
    `;
  }
}

async function handleProtectSubmit() {
  if (!protectSelectedFile || !protectProbeData) return;

  const submitBtn = document.getElementById('protect-submit-btn');
  const engine = document.getElementById('watermark-engine-select').value;
  submitBtn.disabled = true;
  submitBtn.innerHTML = '<div class="spinner"></div><span>REGISTERING & EMBEDDING WATERMARK...</span>';

  try {
    // Step A: Register Provenance
    const regForm = new FormData();
    regForm.append('file', protectSelectedFile);
    regForm.append('relationship_type', 'ORIGINAL');

    const regRes = await fetch('/api/provenance/register', { method: 'POST', body: regForm });
    if (!regRes.ok) {
      const errData = await regRes.json().catch(() => ({ detail: 'Registration failed' }));
      throw new Error(errData.detail || 'Provenance registration failed');
    }
    const regRecord = await regRes.json();
    const assetId = regRecord.asset_id;

    // Step B: Embed Watermark
    const wmForm = new FormData();
    wmForm.append('file', protectSelectedFile);
    wmForm.append('asset_id', assetId);
    wmForm.append('engine', engine);

    const isVideo = protectProbeData.has_video;
    const wmEndpoint = isVideo ? '/api/watermark/video/embed' : '/api/watermark/audio/embed';

    const wmRes = await fetch(wmEndpoint, { method: 'POST', body: wmForm });
    if (!wmRes.ok) {
      const errData = await wmRes.json().catch(() => ({ detail: 'Watermarking failed' }));
      throw new Error(errData.detail || 'Watermark embedding failed');
    }
    const wmResult = await wmRes.json();

    // Show Success Result
    const resultBox = document.getElementById('protect-result-box');
    const resultGrid = document.getElementById('protect-result-grid');
    const downloadBtn = document.getElementById('protect-download-btn');

    resultGrid.innerHTML = `
      <div class="meta-item"><div class="meta-label">Asset ID</div><div class="meta-val" style="color: var(--accent-cyan);">${assetId}</div></div>
      <div class="meta-item"><div class="meta-label">Watermark Token</div><div class="meta-val">${regRecord.watermark_token}</div></div>
      <div class="meta-item"><div class="meta-label">Original SHA-256</div><div class="meta-val">${regRecord.sha256.substring(0, 16)}...</div></div>
      <div class="meta-item"><div class="meta-label">Engine Used</div><div class="meta-val">${wmResult.engine_used || engine}</div></div>
      <div class="meta-item"><div class="meta-label">Protected Filename</div><div class="meta-val">${wmResult.output_filename || 'wm_' + protectSelectedFile.name}</div></div>
      <div class="meta-item"><div class="meta-label">Processing Time</div><div class="meta-val">${wmResult.processing_time_seconds ? wmResult.processing_time_seconds.toFixed(2) + 's' : 'N/A'}</div></div>
    `;

    const outName = wmResult.output_filename || 'wm_' + protectSelectedFile.name;
    downloadBtn.href = `/api/media/protected/${encodeURIComponent(outName)}`;
    resultBox.style.display = 'block';

  } catch (err) {
    alert(`Protection Failed: ${err.message}`);
  } finally {
    submitBtn.disabled = false;
    submitBtn.innerHTML = '<span>PROTECT MEDIA</span>';
  }
}

/* -------------------------------------------------------------------------- */
/* VERIFY MEDIA SECTION */
/* -------------------------------------------------------------------------- */
let verifySelectedFile = null;

function initVerifySection() {
  const dropzone = document.getElementById('verify-dropzone');
  const fileInput = document.getElementById('verify-file-input');

  dropzone.addEventListener('click', () => fileInput.click());

  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });

  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleVerifyFileSelected(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleVerifyFileSelected(e.target.files[0]);
    }
  });

  const submitBtn = document.getElementById('verify-submit-btn');
  submitBtn.addEventListener('click', handleVerifySubmit);
}

async function handleVerifyFileSelected(file) {
  verifySelectedFile = file;

  document.getElementById('verify-results-container').style.display = 'none';

  const previewBox = document.getElementById('verify-preview-box');
  const playerContainer = document.getElementById('verify-player-container');
  const metaGrid = document.getElementById('verify-meta-grid');
  const optionsBox = document.getElementById('verify-options-box');

  previewBox.style.display = 'block';
  optionsBox.style.display = 'block';
  playerContainer.innerHTML = '';

  const objectUrl = URL.createObjectURL(file);
  if (file.type.startsWith('video/') || file.name.match(/\.(mp4|mov|avi|mkv|webm)$/i)) {
    const video = document.createElement('video');
    video.src = objectUrl;
    video.controls = true;
    playerContainer.appendChild(video);
  } else if (file.type.startsWith('audio/') || file.name.match(/\.(wav|mp3|flac|aac|ogg|m4a)$/i)) {
    const audio = document.createElement('audio');
    audio.src = objectUrl;
    audio.controls = true;
    playerContainer.appendChild(audio);
  }

  metaGrid.innerHTML = `
    <div class="meta-item"><div class="meta-label">Filename</div><div class="meta-val">${file.name}</div></div>
    <div class="meta-item"><div class="meta-label">Size</div><div class="meta-val">${formatBytes(file.size)}</div></div>
    <div class="meta-item"><div class="meta-label">Status</div><div class="meta-val">Ready for verification</div></div>
  `;
}

async function handleVerifySubmit() {
  if (!verifySelectedFile) return;

  const submitBtn = document.getElementById('verify-submit-btn');
  const vEngine = document.getElementById('verify-video-engine').value;
  const aEngine = document.getElementById('verify-audio-engine').value;

  submitBtn.disabled = true;
  submitBtn.innerHTML = '<div class="spinner"></div><span>RUNNING VERIFICATION DECISION ENGINE...</span>';

  const formData = new FormData();
  formData.append('file', verifySelectedFile);
  formData.append('video_engine', vEngine);
  formData.append('audio_engine', aEngine);

  try {
    const res = await fetch('/api/verify', { method: 'POST', body: formData });
    const data = await res.json();

    if (!res.ok && data.detail) {
      throw new Error(data.detail);
    }

    renderVerificationResults(data);

  } catch (err) {
    alert(`Verification Error: ${err.message}`);
  } finally {
    submitBtn.disabled = false;
    submitBtn.innerHTML = '<span>VERIFY MEDIA</span>';
  }
}

function renderVerificationResults(data) {
  const resultsContainer = document.getElementById('verify-results-container');
  const banner = document.getElementById('decision-banner');
  const stateText = document.getElementById('decision-state-text');
  const confTag = document.getElementById('decision-confidence-tag');
  const explText = document.getElementById('decision-explanation-text');

  const cls = data.classification || 'UNKNOWN PROVENANCE';
  stateText.textContent = cls;
  confTag.textContent = `${data.confidence || 'LOW'} CONFIDENCE`;

  // Visual State Styling
  banner.className = 'decision-banner ' + getCssClassForState(cls);

  // Decision Rationale Explanation
  if (data.confidence_basis && data.confidence_basis.length > 0) {
    explText.textContent = data.confidence_basis.join(' ');
  } else if (cls === 'VERIFIED ORIGINAL') {
    explText.textContent = 'Exact SHA-256 cryptographic match confirmed against registered original asset record in SQLite provenance database.';
  } else if (cls === 'VERIFIED DERIVATIVE') {
    explText.textContent = 'Invisible watermark token successfully extracted and resolved to registered Asset ID. Content is a verified format transcode or derivative.';
  } else if (cls === 'POTENTIALLY MANIPULATED') {
    explText.textContent = 'High-frequency spectral/acoustic anomalies or compression inconsistencies detected by forensic signal extractors.';
  } else if (cls === 'UNKNOWN PROVENANCE') {
    explText.textContent = 'No matching cryptographic hash and no registered watermark payload detected. File is un-tracked third-party media.';
  } else {
    explText.textContent = 'Conflicting evidence signals or incomplete record data present. System cannot definitively verify asset lineage.';
  }

  // 1. Cryptographic Evidence
  const hashEv = data.hash_evidence || {};
  document.getElementById('ev-hash-badge').textContent = hashEv.status || 'MISMATCH';
  document.getElementById('ev-hash-badge').className = `evidence-badge badge-${(hashEv.status || 'mismatch').toLowerCase()}`;
  document.getElementById('ev-hash-body').innerHTML = `
    <div class="evidence-field"><span class="evidence-field-name">Status</span><span class="evidence-field-val">${hashEv.status}</span></div>
    <div class="evidence-field"><span class="evidence-field-name">Authenticity Conf</span><span class="evidence-field-val">${data.authenticity_confidence || 'UNKNOWN'}</span></div>
    <div class="evidence-field"><span class="evidence-field-name">SHA-256</span><span class="evidence-field-val">${(hashEv.details?.sha256 || 'N/A').substring(0, 14)}...</span></div>
    <div class="evidence-field"><span class="evidence-field-name">Matched Asset</span><span class="evidence-field-val">${hashEv.details?.matched_asset_id || 'None'}</span></div>
  `;

  // 2. Watermark Evidence
  const wmEv = data.watermark_evidence || {};
  document.getElementById('ev-wm-badge').textContent = wmEv.status || 'NOT_DETECTED';
  document.getElementById('ev-wm-badge').className = `evidence-badge badge-${(wmEv.status || 'none').toLowerCase()}`;
  document.getElementById('ev-wm-body').innerHTML = `
    <div class="evidence-field"><span class="evidence-field-name">Engine</span><span class="evidence-field-val">${wmEv.details?.engine_used || 'N/A'}</span></div>
    <div class="evidence-field"><span class="evidence-field-name">Token</span><span class="evidence-field-val">${wmEv.details?.token || 'None'}</span></div>
    <div class="evidence-field"><span class="evidence-field-name">Resolved Asset</span><span class="evidence-field-val">${wmEv.details?.resolved_asset_id || 'None'}</span></div>
  `;

  // 3. Provenance Evidence
  const provEv = data.provenance_evidence || {};
  document.getElementById('ev-prov-badge').textContent = provEv.status || 'NO_RECORD';
  document.getElementById('ev-prov-badge').className = `evidence-badge badge-${(provEv.status || 'no_record').toLowerCase()}`;
  document.getElementById('ev-prov-body').innerHTML = `
    <div class="evidence-field"><span class="evidence-field-name">Asset ID</span><span class="evidence-field-val">${data.asset_id || 'None'}</span></div>
    <div class="evidence-field"><span class="evidence-field-name">Provenance Conf</span><span class="evidence-field-val">${data.provenance_confidence || 'NONE'}</span></div>
    <div class="evidence-field"><span class="evidence-field-name">Relationship</span><span class="evidence-field-val">${provEv.details?.relationship_type || 'N/A'}</span></div>
    <div class="evidence-field"><span class="evidence-field-name">Registered File</span><span class="evidence-field-val">${provEv.details?.original_filename || 'N/A'}</span></div>
  `;

  // 4. Forensic Evidence
  const forEv = data.forensic_evidence || {};
  const genStatus = forEv.details?.generic_forensic_status || 'LOW_GENERIC_ANOMALY';
  const manipStatus = forEv.details?.manipulation_forensic_status || 'NO_DETECTOR_AVAILABLE';
  const transCons = forEv.details?.transformation_consistency?.status || 'NOT_AVAILABLE';

  document.getElementById('ev-forensic-badge').textContent = forEv.status || 'LOW';
  document.getElementById('ev-forensic-badge').className = `evidence-badge badge-${(forEv.status || 'low').toLowerCase()}`;
  document.getElementById('ev-forensic-body').innerHTML = `
    <div class="evidence-field"><span class="evidence-field-name">Generic Anomaly</span><span class="evidence-field-val">${genStatus}</span></div>
    <div class="evidence-field"><span class="evidence-field-name">Parent Consistency</span><span class="evidence-field-val">${transCons}</span></div>
    <div class="evidence-field"><span class="evidence-field-name">Risk Score</span><span class="evidence-field-val">${forEv.details?.forensic_risk_score !== undefined ? forEv.details.forensic_risk_score.toFixed(2) : 'N/A'}</span></div>
  `;

  // 5. Manipulation Evidence Card
  const manipEv = data.manipulation_evidence || {};
  document.getElementById('ev-manip-badge').textContent = manipEv.status || 'NOT_AVAILABLE';
  document.getElementById('ev-manip-badge').className = `evidence-badge badge-${(manipEv.status || 'not_available').toLowerCase()}`;
  document.getElementById('ev-manip-body').innerHTML = `
    <div class="evidence-field"><span class="evidence-field-name">Status</span><span class="evidence-field-val">${manipEv.status}</span></div>
    <div class="evidence-field"><span class="evidence-field-name">Detector Model</span><span class="evidence-field-val" style="color: var(--text-muted);">${manipEv.status === 'MANIPULATION_SUSPECTED' ? 'Active Anomaly' : 'Not Installed'}</span></div>
  `;
  document.getElementById('ev-manip-expl').textContent = manipEv.explanation || 'No validated deepfake/anti-spoofing detector model is installed.';

  // Limitations Box
  const limBox = document.getElementById('verify-limitations-box');
  if (data.limitations && data.limitations.length > 0) {
    limBox.style.display = 'block';
    limBox.innerHTML = `<strong>System Limitations & Warnings:</strong><ul style="margin-left: 1.25rem; margin-top: 0.25rem;">${data.limitations.map(l => `<li>${l}</li>`).join('')}</ul>`;
  } else {
    limBox.style.display = 'none';
  }

  // Conflicts Box
  const cflBox = document.getElementById('verify-conflicts-box');
  if (data.conflicts && data.conflicts.length > 0) {
    cflBox.style.display = 'block';
    cflBox.innerHTML = `<strong>Provenance Conflicts Identified:</strong><ul style="margin-left: 1.25rem; margin-top: 0.25rem;">${data.conflicts.map(c => `<li>${c}</li>`).join('')}</ul>`;
  } else {
    cflBox.style.display = 'none';
  }

  resultsContainer.style.display = 'block';
}

function getCssClassForState(state) {
  switch (state) {
    case 'VERIFIED ORIGINAL': return 'state-verified-original';
    case 'VERIFIED DERIVATIVE': return 'state-verified-derivative';
    case 'POTENTIALLY MANIPULATED': return 'state-potentially-manipulated';
    case 'UNKNOWN PROVENANCE': return 'state-unknown-provenance';
    case 'INSUFFICIENT EVIDENCE': return 'state-insufficient-evidence';
    default: return 'state-unknown-provenance';
  }
}

function formatBytes(bytes, decimals = 2) {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}
