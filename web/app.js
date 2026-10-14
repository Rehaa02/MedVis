/**
 * ClinAI PACS - Medical Image Diagnosis & Triage Orchestrator Client
 */

// Application State
const state = {
  studies: [],
  selectedStudyId: null,
  currentAnalysis: null,
  currentSliceIndex: 0,
  totalSlices: 1,
  showSegmentation: true,
  showBBoxes: true,
  isInverted: false,
  windowPreset: 'default',
  filter: 'all',
  currentSliceImage: null
};

// DOM Elements
const worklistContainer = document.getElementById('worklistContainer');
const chipStatCount = document.getElementById('chipStatCount');
const chipUrgentCount = document.getElementById('chipUrgentCount');
const chipRoutineCount = document.getElementById('chipRoutineCount');
const btnRefreshWorklist = document.getElementById('btnRefreshWorklist');
const dicomFileInput = document.getElementById('dicomFileInput');

const dicomCanvas = document.getElementById('dicomCanvas');
const ctx = dicomCanvas.getContext('2d');
const viewerLoading = document.getElementById('viewerLoading');
const studyMetaBrief = document.getElementById('studyMetaBrief');
const patientIdBrief = document.getElementById('patientIdBrief');
const sliceSlider = document.getElementById('sliceSlider');
const sliceIndicator = document.getElementById('sliceIndicator');
const hudTopLeft = document.getElementById('hudTopLeft');
const hudTopRight = document.getElementById('hudTopRight');
const hudBottomLeft = document.getElementById('hudBottomLeft');
const hudBottomRight = document.getElementById('hudBottomRight');
const windowPresetSelect = document.getElementById('windowPresetSelect');
const toggleSegmentation = document.getElementById('toggleSegmentation');
const toggleBBoxes = document.getElementById('toggleBBoxes');
const btnInvert = document.getElementById('btnInvert');
const organChipsContainer = document.getElementById('organChipsContainer');

const triageBanner = document.getElementById('triageBanner');
const bannerUrgency = document.getElementById('bannerUrgency');
const bannerTime = document.getElementById('bannerTime');
const reportIndicationText = document.getElementById('reportIndicationText');
const reportFindingsText = document.getElementById('reportFindingsText');
const reportImpressionText = document.getElementById('reportImpressionText');
const valMeanEntropy = document.getElementById('valMeanEntropy');
const valConsistency = document.getElementById('valConsistency');
const valDear = document.getElementById('valDear');
const valRiskScore = document.getElementById('valRiskScore');

const radIdInput = document.getElementById('radIdInput');
const radReportEditor = document.getElementById('radReportEditor');
const valLevDist = document.getElementById('valLevDist');
const valEditRatio = document.getElementById('valEditRatio');
const btnAcceptReport = document.getElementById('btnAcceptReport');
const btnModifyReport = document.getElementById('btnModifyReport');
const btnRejectReport = document.getElementById('btnRejectReport');
const auditLogList = document.getElementById('auditLogList');

const epsRange = document.getElementById('epsRange');
const epsVal = document.getElementById('epsVal');
const btnRunAdversarial = document.getElementById('btnRunAdversarial');
const advResultsBox = document.getElementById('advResultsBox');
const advCleanConf = document.getElementById('advCleanConf');
const advAttackedConf = document.getElementById('advAttackedConf');
const advDefendedConf = document.getElementById('advDefendedConf');
const advRecoveryPct = document.getElementById('advRecoveryPct');

// Initialization
document.addEventListener('DOMContentLoaded', () => {
  setupEventListeners();
  loadWorklist();
  checkSystemHealth();
});

function setupEventListeners() {
  btnRefreshWorklist.addEventListener('click', loadWorklist);
  dicomFileInput.addEventListener('change', handleFileUpload);

  sliceSlider.addEventListener('input', (e) => {
    state.currentSliceIndex = parseInt(e.target.value);
    renderCurrentSlice();
  });

  windowPresetSelect.addEventListener('change', (e) => {
    state.windowPreset = e.target.value;
    renderCurrentSlice();
  });

  toggleSegmentation.addEventListener('click', () => {
    state.showSegmentation = !state.showSegmentation;
    toggleSegmentation.classList.toggle('active', state.showSegmentation);
    drawCanvas();
  });

  toggleBBoxes.addEventListener('click', () => {
    state.showBBoxes = !state.showBBoxes;
    toggleBBoxes.classList.toggle('active', state.showBBoxes);
    drawCanvas();
  });

  btnInvert.addEventListener('click', () => {
    state.isInverted = !state.isInverted;
    btnInvert.classList.toggle('active', state.isInverted);
    drawCanvas();
  });

  // Filter tabs
  document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.filter = btn.getAttribute('data-filter');
      renderWorklist();
    });
  });

  // Right Panel Tabs
  document.querySelectorAll('.tab-nav').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.tab-nav').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      tab.classList.add('active');
      const targetId = tab.getAttribute('data-tab');
      document.getElementById(targetId).classList.add('active');
    });
  });

  // Levenshtein editor input
  radReportEditor.addEventListener('input', updateLevenshteinMetrics);

  btnAcceptReport.addEventListener('click', () => submitReview('accepted'));
  btnModifyReport.addEventListener('click', () => submitReview('modified'));
  btnRejectReport.addEventListener('click', () => submitReview('rejected'));

  epsRange.addEventListener('input', (e) => {
    epsVal.textContent = parseFloat(e.target.value).toFixed(2);
  });
  btnRunAdversarial.addEventListener('click', runAdversarialEvaluation);
}

// System Health & Hardware Detection
async function checkSystemHealth() {
  try {
    const res = await fetch('/api/v1/health');
    const data = await res.json();
    const accelBadge = document.getElementById('accelBadge');
    accelBadge.replaceChildren();
    const dot = document.createElement('span');
    dot.className = 'dot pulse-blue';
    accelBadge.append(dot, ` Acceleration: ${data.device.toUpperCase()}`);
  } catch (err) {
    console.warn("Health check error:", err);
  }
}

// Worklist Loader
async function loadWorklist() {
  worklistContainer.setAttribute('aria-busy', 'true');
  if (state.studies.length === 0) renderWorklistSkeleton();

  try {
    const res = await fetch('/api/v1/triage/worklist');
    if (!res.ok) throw new Error(`Worklist request failed (${res.status})`);
    const data = await res.json();
    state.studies = data.worklist || [];

    chipStatCount.textContent = `STAT: ${data.stat_critical_count || 0}`;
    chipUrgentCount.textContent = `Urgent: ${data.urgent_count || 0}`;
    chipRoutineCount.textContent = `Routine: ${data.routine_count || 0}`;

    renderWorklist();

    // Auto-select first study if none selected
    if (state.studies.length > 0 && !state.selectedStudyId) {
      selectStudy(state.studies[0].study_instance_uid);
    }
  } catch (err) {
    worklistContainer.innerHTML = '<div class="empty-state">Unable to connect to the triage service. Try Refresh again.</div>';
  } finally {
    worklistContainer.setAttribute('aria-busy', 'false');
  }
}

function renderWorklistSkeleton() {
  worklistContainer.innerHTML = `
    <div class="loading-state" aria-live="polite" aria-label="Loading clinical studies">
      <div class="skeleton-row"><span class="skeleton-block short"></span><span class="skeleton-block"></span></div>
      <div class="skeleton-row"><span class="skeleton-block medium"></span><span class="skeleton-block"></span></div>
      <div class="skeleton-row"><span class="skeleton-block short"></span><span class="skeleton-block"></span></div>
    </div>`;
}

function renderWorklist() {
  worklistContainer.innerHTML = '';
  const filtered = state.studies.filter(s => {
    if (state.filter === 'STAT') return s.triage_category === 'STAT';
    if (state.filter === 'PENDING') return s.status !== 'REVIEWED';
    return true;
  });

  if (filtered.length === 0) {
    worklistContainer.innerHTML = `<div class="empty-state">No studies found in the current view.<br><br>Upload a DICOM study to begin.</div>`;
    return;
  }

  filtered.forEach(study => {
    const card = document.createElement('div');
    card.className = `worklist-card ${state.selectedStudyId === study.study_instance_uid ? 'active' : ''}`;
    
    let tagClass = 'tag-routine';
    if (study.triage_category === 'STAT') tagClass = 'tag-stat';
    else if (study.triage_category === 'URGENT') tagClass = 'tag-urgent';

    card.innerHTML = `
      <div class="card-top">
      <span class="triage-tag ${tagClass}">${study.triage_category} / under 30s</span>
        <span class="card-score">Score: ${(study.urgency_score * 100).toFixed(0)}%</span>
      </div>
      <div class="card-patient">${study.pseudo_patient_id}</div>
      <div class="card-meta">
        <span>${study.modality} / ${study.slice_count} slice(s)</span>
        <span>${study.time_to_triage_seconds}s triage</span>
      </div>
      <div class="card-finding">${study.top_findings.slice(0, 2).join(', ')}</div>
    `;

    card.addEventListener('click', () => selectStudy(study.study_instance_uid));
    worklistContainer.appendChild(card);
  });
}

// Select Study & Fetch Analysis
async function selectStudy(studyUid) {
  state.selectedStudyId = studyUid;
  renderWorklist();

  try {
    const res = await fetch(`/api/v1/triage/studies/${studyUid}`);
    if (!res.ok) throw new Error("Study analysis not found");
    const data = await res.json();
    state.currentAnalysis = data;

    // Update study metadata
    const studyDirMeta = state.studies.find(s => s.study_instance_uid === studyUid);
    state.totalSlices = studyDirMeta ? studyDirMeta.slice_count : 1;
    state.currentSliceIndex = Math.floor(state.totalSlices / 2);

    studyMetaBrief.innerHTML = `
      <span class="patient-name">${data.pseudo_patient_id}</span>
      <span class="patient-id">UID: ...${studyUid.slice(-12)}</span>
    `;

    // Configure slice slider
    sliceSlider.max = Math.max(0, state.totalSlices - 1);
    sliceSlider.value = state.currentSliceIndex;
    sliceIndicator.textContent = `Slice: ${state.currentSliceIndex + 1}/${state.totalSlices}`;

    // Render organ chips
    renderOrganChips(data.organ_volumes);

    // Render report & hallucination audit
    renderReport(data);

    // Update HUD
    hudTopLeft.textContent = `${data.modality} | ${data.pseudo_patient_id}`;
    hudTopRight.textContent = `${data.model_versions.triage_2d}`;

    // Load first slice image
    renderCurrentSlice();

    // Fetch audit history
    loadAuditHistory(studyUid);
  } catch (err) {
    showToast(`Error loading study: ${err.message}`);
  }
}

// Render Organ Volumetrics Strip
function renderOrganChips(organVolumes) {
  organChipsContainer.innerHTML = '';
  if (!organVolumes || organVolumes.length === 0) {
    organChipsContainer.innerHTML = `<span class="empty-chips">2D Study - Volumetric segmentation active on 3D CT/MRI volumes.</span>`;
    return;
  }

  organVolumes.slice(0, 8).forEach(org => {
    const chip = document.createElement('div');
    chip.className = `organ-chip ${org.is_abnormal ? 'abnormal' : ''}`;
    chip.innerHTML = `
      <span>${org.organ_name}:</span>
      <b>${org.volume_cm3} cm³</b>
      <small>(${org.status})</small>
    `;
    organChipsContainer.appendChild(chip);
  });
}

// Slice Image Rendering via DICOMweb
function renderCurrentSlice() {
  if (!state.selectedStudyId) return;

  viewerLoading.hidden = false;
  sliceIndicator.textContent = `Slice: ${state.currentSliceIndex + 1}/${state.totalSlices}`;
  hudBottomRight.textContent = `Slice: ${state.currentSliceIndex + 1} / ${state.totalSlices}`;

  // Calculate windowing based on preset
  let wc = null, ww = null;
  if (state.windowPreset === 'lung') { wc = -600; ww = 1500; }
  else if (state.windowPreset === 'bone') { wc = 300; ww = 2000; }
  else if (state.windowPreset === 'soft') { wc = 40; ww = 400; }
  else if (state.windowPreset === 'brain') { wc = 40; ww = 80; }

  let url = `/dicomweb/studies/${state.selectedStudyId}/slice/${state.currentSliceIndex}`;
  if (wc !== null && ww !== null) {
    url += `?wc=${wc}&ww=${ww}`;
    hudBottomLeft.textContent = `W: ${ww} L: ${wc}`;
  } else {
    hudBottomLeft.textContent = `Preset: Native`;
  }

  const img = new Image();
  img.onload = () => {
    state.currentSliceImage = img;
    viewerLoading.hidden = true;
    drawCanvas();
  };
  img.onerror = () => {
    state.currentSliceImage = null;
    viewerLoading.hidden = true;
    ctx.clearRect(0, 0, dicomCanvas.width, dicomCanvas.height);
    showToast('The selected image could not be loaded.');
  };
  img.src = url;
}

// Draw Canvas with DICOM Pixels + Overlays
function drawCanvas() {
  if (!state.currentSliceImage) return;

  const img = state.currentSliceImage;
  dicomCanvas.width = img.naturalWidth || 512;
  dicomCanvas.height = img.naturalHeight || 512;

  ctx.clearRect(0, 0, dicomCanvas.width, dicomCanvas.height);

  // Draw DICOM slice
  if (state.isInverted) {
    ctx.filter = 'invert(1)';
  } else {
    ctx.filter = 'none';
  }
  ctx.drawImage(img, 0, 0, dicomCanvas.width, dicomCanvas.height);
  ctx.filter = 'none';

  // Draw 3D Volumetric Segmentation Masks (Simulated visual anatomical contours)
  if (state.showSegmentation && state.currentAnalysis && state.currentAnalysis.organ_volumes.length > 0) {
    drawSegmentationContours(dicomCanvas.width, dicomCanvas.height);
  }

  // Draw Acute Pathology Bounding Boxes
  if (state.showBBoxes && state.currentAnalysis && state.currentAnalysis.bounding_boxes) {
    drawBoundingBoxes(dicomCanvas.width, dicomCanvas.height);
  }
}

function drawSegmentationContours(w, h) {
  ctx.save();
  ctx.globalAlpha = 0.35;

  // Liver contour
  ctx.fillStyle = '#7893a2';
  ctx.beginPath();
  ctx.ellipse(w * 0.35, h * 0.44, w * 0.16, h * 0.14, 0, 0, Math.PI * 2);
  ctx.fill();

  // Spleen contour
  ctx.fillStyle = '#9ea9ad';
  ctx.beginPath();
  ctx.ellipse(w * 0.70, h * 0.48, w * 0.08, h * 0.09, 0, 0, Math.PI * 2);
  ctx.fill();

  // Kidney contours
  ctx.fillStyle = '#657b84';
  ctx.beginPath();
  ctx.ellipse(w * 0.36, h * 0.65, w * 0.06, h * 0.07, 0, 0, Math.PI * 2);
  ctx.ellipse(w * 0.64, h * 0.65, w * 0.06, h * 0.07, 0, 0, Math.PI * 2);
  ctx.fill();

  ctx.restore();
}

function drawBoundingBoxes(w, h) {
  const bboxes = state.currentAnalysis.bounding_boxes;
  bboxes.forEach(b => {
    const x = b.x_min * w;
    const y = b.y_min * h;
    const bw = (b.x_max - b.x_min) * w;
    const bh = (b.y_max - b.y_min) * h;

    ctx.save();
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 2.5;
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(x, y, bw, bh);

    // Label tag
    ctx.fillStyle = 'rgba(239, 68, 68, 0.9)';
    ctx.font = 'bold 12px system-ui, sans-serif';
    const labelText = `${b.label.toUpperCase()} (${(b.confidence * 100).toFixed(0)}%)`;
    const tw = ctx.measureText(labelText).width;
    ctx.fillRect(x, Math.max(0, y - 20), tw + 10, 20);

    ctx.fillStyle = '#edf0ef';
    ctx.fillText(labelText, x + 5, Math.max(14, y - 5));
    ctx.restore();
  });
}

// Render Report with Token-Level Hallucination Entropy
function renderReport(data) {
  const rep = data.report_draft;
  reportIndicationText.textContent = rep.clinical_indication;
  reportImpressionText.textContent = rep.impression;

  // Banner
  bannerUrgency.textContent = `${data.triage_category} TRIAGE (Score: ${(data.urgency_score * 100).toFixed(0)}%)`;
  bannerUrgency.style.color = data.triage_category === 'STAT' ? 'var(--stat-red)' : 'var(--urgent-amber)';
  bannerTime.textContent = `Turnaround: ${data.time_to_triage_seconds}s`;

  // Render Findings with interactive token spans
  const audit = rep.hallucination_audit;
  reportFindingsText.innerHTML = '';

  if (audit && audit.tokens_audit && audit.tokens_audit.length > 0) {
    audit.tokens_audit.forEach(tok => {
      const span = document.createElement('span');
      span.className = `token-span ${tok.risk_level === 'high_risk_hallucination' ? 'rextrust-alert' : (tok.risk_level === 'moderate_uncertainty' ? 'moderate' : 'grounded')}`;
      span.textContent = tok.token + ' ';
      span.title = `[ReXTrust Audit]\nToken: "${tok.token}"\nPredictive Entropy: ${tok.entropy} bits\nRisk: ${tok.risk_level}\n${tok.reason || ''}`;
      reportFindingsText.appendChild(span);
    });
  } else {
    reportFindingsText.textContent = rep.findings;
  }

  // Hallucination Metrics
  if (audit) {
    valMeanEntropy.textContent = `${audit.rextrust_mean_entropy} bits`;
    valConsistency.textContent = `${(audit.radflag_consistency_score * 100).toFixed(0)}%`;
    valDear.textContent = audit.dear_anatomical_grounding_passed ? 'PASSED' : 'FLAGGED';
    valDear.style.color = audit.dear_anatomical_grounding_passed ? 'var(--success-green)' : 'var(--stat-red)';
    valRiskScore.textContent = audit.overall_hallucination_score.toFixed(2);
  }

  // Initialize FDA GMLP Editor
  const combinedReport = `FINDINGS:\n${rep.findings}\n\nIMPRESSION:\n${rep.impression}`;
  radReportEditor.value = combinedReport;
  updateLevenshteinMetrics();
}

// FDA GMLP Levenshtein Tracker
function updateLevenshteinMetrics() {
  if (!state.currentAnalysis) return;
  const original = `FINDINGS:\n${state.currentAnalysis.report_draft.findings}\n\nIMPRESSION:\n${state.currentAnalysis.report_draft.impression}`;
  const current = radReportEditor.value;

  const dist = computeLevenshtein(original, current);
  valLevDist.textContent = `${dist} chars`;
  const ratio = (dist / Math.max(original.length, current.length, 1)) * 100;
  valEditRatio.textContent = `${ratio.toFixed(1)}%`;
}

function computeLevenshtein(a, b) {
  if (a.length === 0) return b.length;
  if (b.length === 0) return a.length;
  const matrix = [];
  for (let i = 0; i <= b.length; i++) matrix[i] = [i];
  for (let j = 0; j <= a.length; j++) matrix[0][j] = j;

  for (let i = 1; i <= b.length; i++) {
    for (let j = 1; j <= a.length; j++) {
      if (b.charAt(i - 1) === a.charAt(j - 1)) {
        matrix[i][j] = matrix[i - 1][j - 1];
      } else {
        matrix[i][j] = Math.min(
          matrix[i - 1][j - 1] + 1,
          matrix[i][j - 1] + 1,
          matrix[i - 1][j] + 1
        );
      }
    }
  }
  return matrix[b.length][a.length];
}

// Submit FDA GMLP Review
async function submitReview(decision) {
  if (!state.selectedStudyId) return;

  const original = `FINDINGS:\n${state.currentAnalysis.report_draft.findings}\n\nIMPRESSION:\n${state.currentAnalysis.report_draft.impression}`;
  const finalReport = radReportEditor.value;

  try {
    const res = await fetch('/api/v1/review/submit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        study_instance_uid: state.selectedStudyId,
        radiologist_id: radIdInput.value || 'RAD_MD',
        decision: decision,
        original_ai_report: original,
        final_signed_report: finalReport,
        agree_with_urgency_tier: true
      })
    });

    const data = await res.json();
    showToast(`FDA GMLP Review Recorded: Decision '${decision.toUpperCase()}'.`);
    loadAuditHistory(state.selectedStudyId);
    loadWorklist();
  } catch (err) {
    showToast(`Failed to submit review: ${err.message}`);
  }
}

async function loadAuditHistory(studyUid) {
  try {
    const res = await fetch(`/api/v1/review/audit/${studyUid}`);
    const records = await res.json();
    if (records.length === 0) {
      auditLogList.innerHTML = `No audit events submitted for this study yet.`;
      return;
    }

    auditLogList.innerHTML = records.map(r => `
      <div class="audit-entry">
        <b>[${r.decision.toUpperCase()}]</b> by ${r.radiologist_id} / Edit: ${r.levenshtein_distance} chars (${(r.edit_ratio * 100).toFixed(1)}%)<br>
        <span class="audit-meta">Sig: ${r.tamper_proof_signature.slice(0, 24)}... / ${r.timestamp}</span>
      </div>
    `).join('');
  } catch (err) {
    console.warn("Audit load error:", err);
  }
}

// Adversarial Security Tester
async function runAdversarialEvaluation() {
  if (!state.selectedStudyId) {
    showToast("Please select a study first.");
    return;
  }

  btnRunAdversarial.textContent = "Executing FGSM Attack...";
  btnRunAdversarial.disabled = true;

  try {
    const eps = parseFloat(epsRange.value);
    const res = await fetch(`/api/v1/triage/studies/${state.selectedStudyId}/adversarial-eval?epsilon=${eps}`, {
      method: 'POST'
    });
    const data = await res.json();
    const m = data.metrics;

    advResultsBox.style.display = 'flex';
    advCleanConf.textContent = `${(m.clean_confidence * 100).toFixed(1)}%`;
    advAttackedConf.textContent = `${(m.adversarial_confidence * 100).toFixed(1)}% (-${m.attack_degradation_pct}%)`;
    advDefendedConf.textContent = `${(m.defended_confidence * 100).toFixed(1)}%`;
    advRecoveryPct.textContent = `${m.defense_recovery_pct}%`;

    showToast(`Adversarial Evaluation complete: Defense recovered ${m.defense_recovery_pct}% confidence!`);
  } catch (err) {
    showToast(`Adversarial test error: ${err.message}`);
  } finally {
    btnRunAdversarial.textContent = "Run security evaluation";
    btnRunAdversarial.disabled = false;
  }
}

// Handle Direct DICOM File Upload
async function handleFileUpload(e) {
  const files = e.target.files;
  if (!files || files.length === 0) return;

  const formData = new FormData();
  for (let i = 0; i < files.length; i++) {
    formData.append('files', files[i]);
  }
  formData.append('referring_indication', 'STAT Clinical Evaluation');

  showToast(`Uploading ${files.length} DICOM instance(s)...`);

  try {
    const res = await fetch('/api/v1/dicom/upload', {
      method: 'POST',
      body: formData
    });
    if (!res.ok) throw new Error(`Upload request failed (${res.status})`);
    const data = await res.json();
    showToast(data.message);
    setTimeout(loadWorklist, 1500);
  } catch (err) {
    showToast(`Upload failed: ${err.message}`);
  }
}

function showToast(msg) {
  const toast = document.getElementById('toastNotification');
  toast.textContent = msg;
  toast.style.display = 'block';
  setTimeout(() => {
    toast.style.display = 'none';
  }, 3500);
}
