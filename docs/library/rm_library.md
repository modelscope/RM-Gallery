---
new: true
---
# RM Library

<div style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.05) 0%, rgba(139, 92, 246, 0.05) 100%); padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 2rem; border: 1px solid rgba(59, 130, 246, 0.15);">
  <h2 style="margin-top: 0; font-size: 1.5rem; color: #1e40af;">📚 Welcome to RM Library</h2>
  <p style="margin-bottom: 0.75rem; line-height: 1.6;">
    Explore our comprehensive collection of <strong>40+ ready-to-use reward models</strong> designed for various evaluation scenarios.
    Our library covers alignment (helpfulness, harmlessness, honesty), code quality, mathematical verification, format validation,
    and general evaluation metrics.
  </p>
  <p style="margin-bottom: 0.75rem; line-height: 1.6;">
    <strong>🎯 What you'll find:</strong>
  </p>
  <ul style="margin-left: 1.5rem; line-height: 1.8;">
    <li><strong>Alignment Models:</strong> 21 models for HHH (Helpfulness, Harmlessness, Honesty) evaluation based on RewardBench2 and RMB Bench</li>
    <li><strong>Code Quality:</strong> 4 models for syntax checking, style validation, patch similarity, and execution testing</li>
    <li><strong>Math Evaluation:</strong> Mathematical expression verification with LaTeX support</li>
    <li><strong>Format & Style:</strong> 5 models for format validation, length control, repetition detection, and privacy protection</li>
    <li><strong>General Metrics:</strong> 4 models including accuracy, F1 score, ROUGE, and numerical accuracy</li>
  </ul>
  <p style="margin-bottom: 0; line-height: 1.6; color: #4b5563;">
    💡 <em>Use the search bar below to find specific models, or browse by category to explore our full collection.</em>
  </p>
</div>

<div id="rm-lib-root" class="ml-prose-container">
  <!-- 工具条 -->
  <div class="ml-card">
    <div class="ml-toolbar">
      <div class="ml-input-wrap">
        <svg class="ml-icon" viewBox="0 0 24 24" aria-hidden="true">
          <path d="M15.5 14h-.79l-.28-.27A6.471 6.471 0 0 0 16 9.5 6.5 6.5 0 1 0 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"/>
        </svg>
        <input id="rm-search" placeholder="Search reward models..." />
      </div>
      <button id="rm-clear" class="ml-btn secondary">Clear</button>
    </div>
    <div id="rm-stats" class="ml-stats" hidden>
      <span>Showing <b id="rm-count">0</b> of <b id="rm-total">0</b> <span id="rm-type">reward models</span></span>
    </div>
  </div>

  <!-- 加载/错误 -->
  <div id="rm-loading" class="ml-loading">
    <div class="ml-spinner" aria-label="Loading"></div>
    <div class="ml-muted">Loading RM library…</div>
  </div>
  <div id="rm-error" class="ml-error" hidden>
    <div class="ml-error-icon">⚠️</div>
    <div class="ml-muted">Failed to load RM library.</div>
    <button id="rm-retry" class="ml-btn">Try again</button>
  </div>

  <!-- 面包屑 -->
  <div id="rm-crumb" class="ml-crumb" hidden>
    <button id="rm-back" class="ml-link">← Back to Categories</button>
    <div class="ml-crumb-title" id="rm-crumb-title">RM Categories</div>
  </div>

  <!-- 列表容器 -->
  <div id="rm-categories" class="ml-stacked" hidden></div>
  <div id="rm-models" class="ml-grid" hidden></div>

  <!-- 空态 -->
  <div id="rm-empty" class="ml-empty" hidden>
    <div class="ml-empty-icon">🔎</div>
    <div class="ml-muted">No reward models found. Try changing your search.</div>
  </div>
</div>

<!-- 详情弹窗 -->
<dialog id="rm-modal" class="ml-modal">
  <form method="dialog" class="ml-modal-card">
    <div class="ml-modal-header">
      <div>
        <div class="ml-chip" id="rm-modal-category"></div>
        <div class="ml-chip success" id="rm-modal-type"></div>
      </div>
    </div>

    <div class="ml-modal-section">
      <div class="ml-section-title">Description</div>
      <div class="ml-note" id="rm-modal-description"></div>
    </div>

    <div class="ml-modal-section">
      <div class="ml-section-title">Scenario</div>
      <div class="ml-code" id="rm-modal-scenario"></div>
    </div>

    <div class="ml-modal-section" id="rm-principles-section">
      <div class="ml-section-title">Evaluation Principles</div>
      <div id="rm-modal-principles"></div>
    </div>

    <div class="ml-modal-section">
      <div class="ml-section-title">Usage Example</div>
      <div class="ml-code" id="rm-modal-usage"></div>
    </div>

    <div class="ml-modal-section">
      <div class="ml-section-title">Registry Information</div>
      <div class="ml-meta">
        <div><span>Registry Name</span><b id="rm-modal-registry" class="mono"></b></div>
        <div><span>Class Name</span><b id="rm-modal-class" class="mono"></b></div>
        <div><span>Module Path</span><b id="rm-modal-module" class="mono"></b></div>
        <div><span>Reward Type</span><b id="rm-modal-reward-type"></b></div>
      </div>
    </div>

    <div class="ml-modal-footer">
      <button class="ml-btn secondary" value="cancel">Close</button>
    </div>
  </form>
</dialog>

<style>
:root {
  --ml-radius: .75rem;
  --ml-gap: 1.25rem;
  --ml-shadow: 0 6px 24px rgba(0,0,0,.08);
}
.ml-prose-container { display: grid; gap: var(--ml-gap); }
.ml-card {
  background: var(--background, #fff);
  color: var(--foreground, #0a0a0a);
  border: 1px solid var(--border, rgba(0,0,0,.08));
  border-radius: var(--ml-radius);
  padding: 1rem;
  box-shadow: var(--shadow, 0 1px 0 rgba(0,0,0,.02));
}

/* general card/grid */
.ml-grid {
  display: grid;
  gap: var(--ml-gap);
  grid-template-columns: repeat(1, minmax(0,1fr));
}
@media (min-width: 768px){ .ml-grid{ grid-template-columns: repeat(2, minmax(0,1fr)); } }
@media (min-width: 1400px){ .ml-grid{ grid-template-columns: repeat(3, minmax(0,1fr)); } }

/* categories stacked */
.ml-stacked { display: grid; gap: 1.25rem; }
.ml-section{ display:grid; gap:1rem; margin-bottom: 2rem; }
.ml-section h3{
  margin:.5rem 0 1rem 0;
  font-size:1.25rem;
  font-weight:700;
  opacity:.9;
  display:flex;
  gap:.5rem;
  align-items:center;
  border-bottom: 2px solid var(--border, rgba(0,0,0,.08));
  padding-bottom: .75rem;
}
.ml-section-icon { font-size: 1.3rem; }
.ml-section-count {
  margin-left: auto;
  font-size: .85rem;
  font-weight: 500;
  opacity: .6;
  background: var(--muted, rgba(0,0,0,.04));
  padding: .25rem .65rem;
  border-radius: .4rem;
}

.ml-card-item{
  background: var(--card, var(--background, #fff));
  border: 1px solid var(--border, rgba(0,0,0,.08));
  border-radius: var(--ml-radius);
  padding: 1.25rem;
  transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease;
  cursor: pointer;
  box-shadow: 0 1px 3px rgba(0,0,0,.04);
  position: relative;
  overflow: hidden;
  min-height: 200px;
  display: flex;
  flex-direction: column;
}
.ml-card-item:hover{
  transform: translateY(-3px);
  box-shadow: 0 8px 24px rgba(0,0,0,.1), 0 2px 8px rgba(0,0,0,.06);
  border-color: var(--primary, #3b82f6);
}
.ml-card-item:active{
  transform: translateY(-1px);
}
.ml-card-head{ display:flex; align-items:flex-start; justify-content:space-between; gap:.5rem; margin-bottom:.75rem; }
.ml-card-head > div:first-child { flex: 1; min-width: 0; }
.ml-card-left { display: flex; gap: .4rem; flex-wrap: nowrap; align-items: center; }
.ml-card-title{ font-weight: 650; font-size: 1rem; line-height: 1.4; }
.ml-card-title-main {
  font-weight: 650;
  font-size: 1.05rem;
  line-height: 1.4;
  margin-bottom: .35rem;
  color: var(--foreground, #0a0a0a);
  word-break: break-word;
  overflow-wrap: break-word;
  hyphens: auto;
}
.ml-card-class {
  font-size: .8rem;
  opacity: .65;
  margin-bottom: .65rem;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  color: var(--muted-foreground, #6b7280);
  word-break: break-word;
  overflow-wrap: break-word;
}
.ml-card-sub{ font-size: .85rem; opacity: .7; margin-top: .2rem; }
.ml-card-sample{
  margin-top:.5rem;
  font-size:.9rem;
  line-height:1.55;
  opacity:.85;
  display:-webkit-box;
  -webkit-line-clamp:3;
  -webkit-box-orient:vertical;
  overflow:hidden;
  color: var(--muted-foreground, #4b5563);
  flex-grow: 1;
}
.ml-card-foot{
  display:flex;
  justify-content:space-between;
  align-items:center;
  border-top:1px solid var(--border, rgba(0,0,0,.06));
  padding-top:.65rem;
  margin-top:.85rem;
  font-size:.82rem;
}

/* toolbar */
.ml-toolbar{ display:flex; gap:.75rem; align-items:center; justify-content:space-between; flex-wrap:wrap; }
.ml-input-wrap{ position:relative; flex:1; min-width: 260px; }
.ml-input-wrap input{
  width:100%; padding:.6rem .9rem .6rem 2.2rem; border-radius:.6rem;
  border:1px solid var(--border, rgba(0,0,0,.12));
  background: var(--muted, rgba(0,0,0,.02));
  color: var(--foreground, #0a0a0a);
  outline:none;
}
.ml-input-wrap input:focus{
  border-color: var(--primary, #3b82f6);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--primary, #3b82f6) 22%, transparent);
  background: var(--background, #fff);
}
.ml-icon{ position:absolute; left:.6rem; top:50%; transform:translateY(-50%); width:1.1rem; height:1.1rem; opacity:.6; }

.ml-btn{
  border:1px solid var(--border, rgba(0,0,0,.12));
  background: var(--accent, var(--background, #fff));
  color: var(--foreground, #0a0a0a);
  padding:.55rem 1.2rem;
  border-radius:.55rem;
  cursor:pointer;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  font-weight: 500;
  position: relative;
  overflow: hidden;
}
.ml-btn.secondary{
  background: linear-gradient(135deg, var(--primary, #3b82f6) 0%, color-mix(in srgb, var(--primary, #3b82f6) 90%, #6366f1) 100%);
  color: #fff;
  border: none;
  box-shadow: 0 2px 4px rgba(59, 130, 246, 0.2);
}
.ml-btn.secondary:hover{
  background: linear-gradient(135deg, color-mix(in srgb, var(--primary, #3b82f6) 90%, #000) 0%, color-mix(in srgb, var(--primary, #3b82f6) 80%, #000) 100%);
  box-shadow: 0 6px 16px rgba(59, 130, 246, 0.4), 0 2px 8px rgba(59, 130, 246, 0.2);
  transform: translateY(-2px) scale(1.02);
}
.ml-btn.secondary:active{
  transform: translateY(0) scale(0.98);
  box-shadow: 0 2px 4px rgba(59, 130, 246, 0.3);
}
.ml-btn:hover{ border-color: var(--primary, #3b82f6); }

/* 支持减少动画偏好设置 */
@media (prefers-reduced-motion: reduce) {
  .ml-btn {
    transition: none;
  }
  .ml-btn.secondary:hover {
    transform: none;
  }
}

/* stats/breadcrumb */
.ml-stats{ margin-top:.5rem; font-size:.9rem; opacity:.8; }
.ml-crumb{ display:flex; align-items:center; gap:.75rem; }
.ml-link{ background:none; border:none; color: var(--primary, #3b82f6); cursor:pointer; padding:.25rem .5rem; border-radius:.4rem; }
.ml-link:hover{ text-decoration: underline; }
.ml-crumb-title{ font-weight:600; opacity:.8; }

/* states */
.ml-loading, .ml-error, .ml-empty{ display:grid; justify-items:center; gap:.5rem; padding:3rem 1rem; }
.ml-spinner{
  width:38px; height:38px; border-radius:999px; border:3px solid color-mix(in srgb, var(--foreground,#000) 12%, transparent);
  border-top-color: var(--primary,#3b82f6); animation: ml-spin 1s linear infinite;
}
@keyframes ml-spin{ to{ transform: rotate(360deg); } }
.ml-muted{ opacity:.7; }
.ml-error-icon{ font-size:1.4rem; }

/* chips */
.ml-chip{
  display:inline-block;
  padding:.18rem .45rem;
  border-radius:999px;
  font-size:.62rem;
  background: color-mix(in srgb, var(--primary,#3b82f6) 12%, transparent);
  color: var(--primary,#3b82f6);
  white-space: nowrap;
  flex-shrink: 0;
  font-weight: 600;
  letter-spacing: 0;
  text-transform: uppercase;
  border: 1px solid color-mix(in srgb, var(--primary,#3b82f6) 20%, transparent);
  line-height: 1.3;
}
.ml-chip.success{
  background: color-mix(in srgb, #16a34a 12%, transparent);
  color: #16a34a;
  border-color: color-mix(in srgb, #16a34a 20%, transparent);
}
.ml-chip.warning{
  background: color-mix(in srgb, #f59e0b 12%, transparent);
  color: #d97706;
  border-color: color-mix(in srgb, #f59e0b 20%, transparent);
}
.ml-chip.alignment-base {
  background: color-mix(in srgb, #8b5cf6 12%, transparent);
  color: #7c3aed;
  border-color: color-mix(in srgb, #8b5cf6 20%, transparent);
}
.ml-chip.alignment-helpfulness {
  background: color-mix(in srgb, #3b82f6 12%, transparent);
  color: #1d4ed8;
  border-color: color-mix(in srgb, #3b82f6 20%, transparent);
}
.ml-chip.alignment-harmlessness {
  background: color-mix(in srgb, #ef4444 12%, transparent);
  color: #dc2626;
  border-color: color-mix(in srgb, #ef4444 20%, transparent);
}
.ml-chip.alignment-honesty {
  background: color-mix(in srgb, #10b981 12%, transparent);
  color: #059669;
  border-color: color-mix(in srgb, #10b981 20%, transparent);
}
.ml-chip.code {
  background: color-mix(in srgb, #06b6d4 12%, transparent);
  color: #0891b2;
  border-color: color-mix(in srgb, #06b6d4 20%, transparent);
}
.ml-chip.math {
  background: color-mix(in srgb, #f59e0b 12%, transparent);
  color: #d97706;
  border-color: color-mix(in srgb, #f59e0b 20%, transparent);
}
.ml-chip.format {
  background: color-mix(in srgb, #ec4899 12%, transparent);
  color: #db2777;
  border-color: color-mix(in srgb, #ec4899 20%, transparent);
}
.ml-chip.general {
  background: color-mix(in srgb, #6b7280 12%, transparent);
  color: #4b5563;
  border-color: color-mix(in srgb, #6b7280 20%, transparent);
}

/* code/note */
.ml-code{
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
  background: var(--muted, rgba(0,0,0,.04)); border:1px solid var(--border, rgba(0,0,0,.08));
  padding:.75rem; border-radius:.6rem; white-space:pre-wrap;
}
.ml-note{
  background: color-mix(in srgb, #3b82f6 9%, transparent);
  border:1px solid color-mix(in srgb, #3b82f6 28%, transparent);
  padding:.75rem; border-radius:.6rem;
}

/* principles list */
.principle-list {
  list-style: none;
  padding: 0;
  margin: 0;
}
.principle-item {
  background: var(--muted, rgba(0,0,0,.04));
  border: 1px solid var(--border, rgba(0,0,0,.08));
  border-radius: .5rem;
  padding: .75rem;
  margin: .5rem 0;
}
.principle-number {
  font-weight: 600;
  color: var(--primary, #3b82f6);
  margin-right: .5rem;
}

/* meta */
.ml-meta{ display:grid; grid-template-columns: repeat(1, minmax(0,1fr)); gap:.5rem; }
@media (min-width: 640px){ .ml-meta{ grid-template-columns: repeat(2, minmax(0,1fr)); } }
.ml-meta > div{ display:flex; justify-content:space-between; align-items:center; padding:.5rem .75rem;
  border:1px dashed var(--border, rgba(0,0,0,.12)); border-radius:.5rem; background: var(--background, #fff);
  gap: .5rem;
}
.ml-meta span{ opacity:.7; flex-shrink: 0; }
.ml-meta b{
  word-break: break-all;
  overflow-wrap: break-word;
  text-align: right;
  min-width: 0;
  max-width: 100%;
}
.mono{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }

/* modal */
.ml-modal{ padding:0; border:none; background: transparent; }
.ml-modal[open]{ display:grid; align-items:center; justify-items:center; }
.ml-modal::backdrop{ background: rgba(0,0,0,.45); }
.ml-modal-card{
  width:min(100%, 960px); max-height: 85vh; overflow:auto;
  background: var(--background, #fff); color: var(--foreground,#0a0a0a);
  border:1px solid var(--border, rgba(0,0,0,.1)); border-radius: var(--ml-radius);
  padding: 1rem; box-shadow: var(--ml-shadow);
}
.ml-modal-header{ display:flex; justify-content:flex-start; align-items:center; gap:.75rem; margin-bottom:.5rem; }
.ml-modal-section{ display:grid; gap:.35rem; margin-top:.75rem; }
.ml-section-title{ font-weight:650; opacity:.85; }
.ml-modal-footer{ display:flex; justify-content:flex-end; margin-top:1rem; }
</style>

<script>
(() => {
  // —— State
  let ALL_RMS = [];
  let GROUPED_RMS = {};
  let VIEW = "categories"; // "categories" | "models"
  let CURR_CATEGORY = null;

  // —— DOM
  const $ = (id) => document.getElementById(id);
  const elLoading = $("rm-loading");
  const elError = $("rm-error");
  const elRetry = $("rm-retry");
  const elCategories = $("rm-categories");
  const elModels = $("rm-models");
  const elEmpty = $("rm-empty");
  const elSearch = $("rm-search");
  const elClear = $("rm-clear");
  const elStats = $("rm-stats");
  const elCount = $("rm-count");
  const elTotal = $("rm-total");
  const elType = $("rm-type");
  const elCrumb = $("rm-crumb");
  const elBack = $("rm-back");
  const elCrumbTitle = $("rm-crumb-title");
  const dlg = $("rm-modal");

  // Modal elements
  const mCategory = $("rm-modal-category");
  const mType = $("rm-modal-type");
  const mDescription = $("rm-modal-description");
  const mScenario = $("rm-modal-scenario");
  const mPrinciples = $("rm-modal-principles");
  const mUsage = $("rm-modal-usage");
  const mRegistry = $("rm-modal-registry");
  const mClass = $("rm-modal-class");
  const mModule = $("rm-modal-module");
  const mRewardType = $("rm-modal-reward-type");
  const principlesSection = $("rm-principles-section");

  // —— Categories Configuration
  const CATEGORY_MAP = {
    "Alignment - Helpfulness": ["alignment-helpfulness"],
    "Alignment - Harmlessness": ["alignment-harmlessness"],
    "Alignment - Honesty": ["alignment-honesty"],
    "Alignment - Base": ["alignment-base"],
    "Code Quality": ["code"],
    "Math Evaluation": ["math"],
    "Format & Style": ["format"],
    "General Evaluation": ["general"]
  };

  // Category display names for chips (shortened versions)
  const CATEGORY_CHIP_NAMES = {
    "alignment-helpfulness": "HELPFULNESS",
    "alignment-harmlessness": "HARMLESSNESS",
    "alignment-honesty": "HONESTY",
    "alignment-base": "BASE",
    "code": "CODE",
    "math": "MATH",
    "format": "FORMAT",
    "general": "GENERAL"
  };

  // —— RM Data - Complete Reward Model Library
  const MOCK_RMS = [
    // ============= Alignment - Base Models =============
    {
      name: "base_helpfulness_pointwise",
      class_name: "BaseHelpfulnessPointwiseReward",
      category: "alignment-base",
      reward_type: "PointWise",
      description: "Base helpfulness evaluator for assessing how helpful and informative responses are to users with relevant and accurate information.",
      scenario: "General purpose helpfulness assessment for individual responses",
      principles: ["Provide helpful and informative responses", "Respond with relevant and accurate information", "Address user queries effectively"],
      module_path: "rm_gallery.gallery.rm.alignment.base"
    },
    {
      name: "base_helpfulness_listwise",
      class_name: "BaseHelpfulnessListwiseReward",
      category: "alignment-base",
      reward_type: "ListWise",
      description: "Base helpfulness evaluator for comparing multiple responses based on their helpfulness and informativeness.",
      scenario: "Comparative evaluation of response helpfulness",
      principles: ["Provide helpful and informative responses", "Respond with relevant and accurate information", "Address user queries effectively"],
      module_path: "rm_gallery.gallery.rm.alignment.base"
    },
    {
      name: "base_harmlessness_pointwise",
      class_name: "BaseHarmlessnessPointwiseReward",
      category: "alignment-base",
      reward_type: "PointWise",
      description: "Base harmlessness evaluator for detecting harmful behaviors such as spreading misinformation or engaging in harmful activities.",
      scenario: "Safety assessment for individual responses",
      principles: ["Avoid harmful behaviors", "Prevent spreading misinformation", "Refuse harmful requests appropriately"],
      module_path: "rm_gallery.gallery.rm.alignment.base"
    },
    {
      name: "base_harmlessness_listwise",
      class_name: "BaseHarmlessnessListwiseReward",
      category: "alignment-base",
      reward_type: "ListWise",
      description: "Base harmlessness evaluator for comparing multiple responses based on safety and harmlessness criteria.",
      scenario: "Comparative safety assessment",
      principles: ["Avoid harmful behaviors", "Prevent spreading misinformation", "Refuse harmful requests appropriately"],
      module_path: "rm_gallery.gallery.rm.alignment.base"
    },
    {
      name: "base_honesty_pointwise",
      class_name: "BaseHonestyPointwiseReward",
      category: "alignment-base",
      reward_type: "PointWise",
      description: "Base honesty evaluator for assessing truthfulness and bias-free responses.",
      scenario: "Truthfulness assessment for individual responses",
      principles: ["Truthfully answer questions", "Avoid bias and prejudice", "Present information honestly"],
      module_path: "rm_gallery.gallery.rm.alignment.base"
    },
    {
      name: "base_honesty_listwise",
      class_name: "BaseHonestyListwiseReward",
      category: "alignment-base",
      reward_type: "ListWise",
      description: "Base honesty evaluator for comparing multiple responses based on truthfulness and honesty.",
      scenario: "Comparative honesty assessment",
      principles: ["Truthfully answer questions", "Avoid bias and prejudice", "Present information honestly"],
      module_path: "rm_gallery.gallery.rm.alignment.base"
    },

    // ============= Alignment - Harmlessness =============
    {
      name: "safety_pointwise_reward",
      class_name: "SafetyPointwiseReward",
      category: "alignment-harmlessness",
      reward_type: "PointWise",
      description: "Safety evaluator for assessing compliance with or refusal of prompts related to harmful use cases and general compliance behaviors.",
      scenario: "Safety assessment in response to potentially harmful requests (from RewardBench2)",
      principles: ["Refuse harmful prompts appropriately", "Comply with safe requests", "Identify potential risks in user requests"],
      module_path: "rm_gallery.gallery.rm.alignment.harmlessness.safety"
    },
    {
      name: "DetoxifyReward",
      class_name: "DetoxifyReward",
      category: "alignment-harmlessness",
      reward_type: "PointWise",
      description: "Detoxify-based toxicity detector for identifying threats, obscenity, insults, and various types of toxic content.",
      scenario: "Content moderation and toxicity detection across various text types",
      principles: null,
      module_path: "rm_gallery.gallery.rm.alignment.harmlessness.detoxify"
    },

    // ============= Alignment - Helpfulness =============
    {
      name: "brainstorming_listwise_reward",
      class_name: "BrainstormingListwiseReward",
      category: "alignment-helpfulness",
      reward_type: "ListWise",
      description: "Evaluates text generation for brainstorming, emphasizing creativity and driving thinking to come up with new ideas or solutions.",
      scenario: "Creative ideation and brainstorming tasks (from RMB Bench)",
      principles: null,
      module_path: "rm_gallery.gallery.rm.alignment.helpfulness.brainstorming"
    },
    {
      name: "chat_listwise_reward",
      class_name: "ChatListwiseReward",
      category: "alignment-helpfulness",
      reward_type: "ListWise",
      description: "Chat evaluator that simulates human conversation, emphasizing coherence and natural flow of interaction across various topics.",
      scenario: "Conversational AI evaluation with focus on natural dialogue (from RMB Bench)",
      principles: [
        "Address Core Argument/Intent Directly: Prioritize engaging with the user's central claim, perspective, or question explicitly.",
        "Provide Actionable, Context-Specific Guidance: Offer concrete, practical steps tailored to the user's unique situation.",
        "Ensure Factual Accuracy and Contextual Nuance: Ground responses in precise details while avoiding oversimplification."
      ],
      module_path: "rm_gallery.gallery.rm.alignment.helpfulness.chat"
    },
    {
      name: "classification_listwise_reward",
      class_name: "ClassificationListwiseReward",
      category: "alignment-helpfulness",
      reward_type: "ListWise",
      description: "Evaluates classification tasks that assign predefined categories or labels to text based on its content.",
      scenario: "Text classification and categorization tasks (from RMB Bench)",
      principles: null,
      module_path: "rm_gallery.gallery.rm.alignment.helpfulness.classification"
    },
    {
      name: "closed_qa_listwise_reward",
      class_name: "ClosedQAListwiseReward",
      category: "alignment-helpfulness",
      reward_type: "ListWise",
      description: "Evaluates closed QA tasks where answers are found in given context or options, focusing on accuracy within constraints.",
      scenario: "Closed-domain question answering with given context (from RMB Bench)",
      principles: null,
      module_path: "rm_gallery.gallery.rm.alignment.helpfulness.closed_qa"
    },
    {
      name: "code_listwise_reward",
      class_name: "CodeListwiseReward",
      category: "alignment-helpfulness",
      reward_type: "ListWise",
      description: "Evaluates code generation, understanding, and modification tasks within text.",
      scenario: "Programming code generation and comprehension (from RMB Bench)",
      principles: null,
      module_path: "rm_gallery.gallery.rm.alignment.helpfulness.code"
    },
    {
      name: "generation_listwise_reward",
      class_name: "GenerationListwiseReward",
      category: "alignment-helpfulness",
      reward_type: "ListWise",
      description: "Evaluates creative text generation from articles to stories, emphasizing originality and creativity.",
      scenario: "Creative content generation tasks (from RMB Bench)",
      principles: ["Demonstrate originality", "Show creativity in content", "Maintain coherent narrative"],
      module_path: "rm_gallery.gallery.rm.alignment.helpfulness.generation"
    },
    {
      name: "open_qa_listwise_reward",
      class_name: "OpenQAListwiseReward",
      category: "alignment-helpfulness",
      reward_type: "ListWise",
      description: "Evaluates open-domain question answering across wide text sources, requiring processing of large information and complex questions.",
      scenario: "Open-domain question answering without given context (from RMB Bench)",
      principles: null,
      module_path: "rm_gallery.gallery.rm.alignment.helpfulness.open_qa"
    },
    {
      name: "reasoning_listwise_reward",
      class_name: "ReasoningListwiseReward",
      category: "alignment-helpfulness",
      reward_type: "ListWise",
      description: "Evaluates reasoning tasks involving text analysis to draw inferences, make predictions, or solve problems.",
      scenario: "Logical reasoning and inference tasks (from RMB Bench)",
      principles: null,
      module_path: "rm_gallery.gallery.rm.alignment.helpfulness.reasoning"
    },
    {
      name: "rewrite_listwise_reward",
      class_name: "RewriteListwiseReward",
      category: "alignment-helpfulness",
      reward_type: "ListWise",
      description: "Evaluates text rewriting that modifies style while preserving original information and intent.",
      scenario: "Text rewriting and paraphrasing tasks (from RMB Bench)",
      principles: null,
      module_path: "rm_gallery.gallery.rm.alignment.helpfulness.rewrite"
    },
    {
      name: "role_playing_listwise_reward",
      class_name: "RolePlayingListwiseReward",
      category: "alignment-helpfulness",
      reward_type: "ListWise",
      description: "Evaluates role-playing scenarios where AI adopts specific characters or personas in text-based interactions.",
      scenario: "Character role-playing and persona adoption (from RMB Bench)",
      principles: ["Maintain character consistency", "Engage authentically in role", "Reflect assigned persona accurately"],
      module_path: "rm_gallery.gallery.rm.alignment.helpfulness.role_playing"
    },
    {
      name: "summarization_listwise_reward",
      class_name: "SummarizationListwiseReward",
      category: "alignment-helpfulness",
      reward_type: "ListWise",
      description: "Evaluates text summarization that compresses content into short form while retaining main information.",
      scenario: "Text summarization and compression tasks (from RMB Bench)",
      principles: ["Retain key information", "Maintain coherence", "Achieve appropriate compression"],
      module_path: "rm_gallery.gallery.rm.alignment.helpfulness.summarization"
    },
    {
      name: "translation_listwise_reward",
      class_name: "TranslationListwiseReward",
      category: "alignment-helpfulness",
      reward_type: "ListWise",
      description: "Evaluates translation quality for converting text from one language to another.",
      scenario: "Language translation tasks (from RMB Bench)",
      principles: ["Preserve original meaning", "Maintain natural language flow", "Consider cultural context"],
      module_path: "rm_gallery.gallery.rm.alignment.helpfulness.translation"
    },
    {
      name: "focus_pointwise_reward",
      class_name: "FocusPointwiseReward",
      category: "alignment-helpfulness",
      reward_type: "PointWise",
      description: "Detects high-quality, on-topic answers to general user queries with strong focus on the question.",
      scenario: "Evaluating response relevance and focus (from RMB Bench)",
      principles: ["Stay on topic", "Address the question directly", "Avoid tangential information"],
      module_path: "rm_gallery.gallery.rm.alignment.helpfulness.focus"
    },
    {
      name: "math_pointwise_reward",
      class_name: "MathPointwiseReward",
      category: "alignment-helpfulness",
      reward_type: "PointWise",
      description: "Evaluates mathematical problem-solving from middle school to college level, including physics, geometry, calculus, and more.",
      scenario: "Mathematical problem solving across difficulty levels (from RewardBench2)",
      principles: ["Show clear reasoning", "Apply correct formulas", "Verify calculations"],
      module_path: "rm_gallery.gallery.rm.alignment.helpfulness.math"
    },
    {
      name: "precise_if_pointwise_reward",
      class_name: "PreciseIFPointwiseReward",
      category: "alignment-helpfulness",
      reward_type: "PointWise",
      description: "Evaluates precise instruction following with specific constraints like 'Answer without the letter u'.",
      scenario: "Precise instruction following with explicit constraints (from RewardBench2)",
      principles: ["Follow all specified constraints", "Maintain response quality", "Demonstrate attention to detail"],
      module_path: "rm_gallery.gallery.rm.alignment.helpfulness.precise_if"
    },

    // ============= Alignment - Honesty =============
    {
      name: "factuality_pointwise_reward",
      class_name: "FactualityPointwiseReward",
      category: "alignment-honesty",
      reward_type: "PointWise",
      description: "Detects hallucinations and basic errors in completions, ensuring factual accuracy.",
      scenario: "Factuality verification and hallucination detection (from RewardBench2)",
      principles: ["Verify factual claims", "Identify hallucinations", "Ensure accuracy"],
      module_path: "rm_gallery.gallery.rm.alignment.honesty.factuality"
    },

    // ============= Math Evaluation =============
    {
      name: "math_verify_reward",
      class_name: "MathVerifyReward",
      category: "math",
      reward_type: "PointWise",
      description: "Verifies mathematical expressions using the math_verify library, supporting both LaTeX and plain expressions.",
      scenario: "Mathematical expression verification and validation",
      principles: null,
      module_path: "rm_gallery.gallery.rm.math.math"
    },

    // ============= Code Quality =============
    {
      name: "code_syntax_check",
      class_name: "SyntaxCheckReward",
      category: "code",
      reward_type: "PointWise",
      description: "Checks code syntax using Abstract Syntax Tree (AST) to validate Python code blocks for syntax errors.",
      scenario: "Python code syntax validation",
      principles: null,
      module_path: "rm_gallery.gallery.rm.code.code"
    },
    {
      name: "code_style",
      class_name: "CodeStyleReward",
      category: "code",
      reward_type: "PointWise",
      description: "Performs basic code style checking including indentation consistency and naming convention validation.",
      scenario: "Python code style and formatting assessment",
      principles: null,
      module_path: "rm_gallery.gallery.rm.code.code"
    },
    {
      name: "code_patch_similarity",
      class_name: "PatchSimilarityReward",
      category: "code",
      reward_type: "PointWise",
      description: "Calculates similarity between generated patch and oracle patch using difflib.SequenceMatcher.",
      scenario: "Code patch comparison and similarity measurement",
      principles: null,
      module_path: "rm_gallery.gallery.rm.code.code"
    },
    {
      name: "code_execution",
      class_name: "CodeExecutionReward",
      category: "code",
      reward_type: "PointWise",
      description: "Executes code against test cases and evaluates correctness based on test results.",
      scenario: "Functional correctness testing for generated code",
      principles: null,
      module_path: "rm_gallery.gallery.rm.code.code"
    },

    // ============= General Evaluation =============
    {
      name: "accuracy",
      class_name: "AccuracyReward",
      category: "general",
      reward_type: "PointWise",
      description: "Calculates accuracy (exact match rate) between generated content and reference answer.",
      scenario: "Exact match evaluation for classification and QA tasks",
      principles: null,
      module_path: "rm_gallery.gallery.rm.general.general"
    },
    {
      name: "f1_score",
      class_name: "F1ScoreReward",
      category: "general",
      reward_type: "PointWise",
      description: "Calculates F1 score between generated content and reference answer at word level with configurable tokenizer.",
      scenario: "Token-level evaluation for text generation quality",
      principles: null,
      module_path: "rm_gallery.gallery.rm.general.general"
    },
    {
      name: "rouge",
      class_name: "RougeReward",
      category: "general",
      reward_type: "PointWise",
      description: "ROUGE-L similarity evaluation using longest common subsequence for text overlap measurement.",
      scenario: "Summarization and text generation overlap evaluation",
      principles: null,
      module_path: "rm_gallery.gallery.rm.general.general"
    },
    {
      name: "number_accuracy",
      class_name: "NumberAccuracyReward",
      category: "general",
      reward_type: "PointWise",
      description: "Checks numerical calculation accuracy by comparing numbers in generated content versus reference.",
      scenario: "Numerical accuracy verification in mathematical and quantitative tasks",
      principles: null,
      module_path: "rm_gallery.gallery.rm.general.general"
    },

    // ============= Format & Style =============
    {
      name: "reasoning_format",
      class_name: "ReasoningFormatReward",
      category: "format",
      reward_type: "PointWise",
      description: "Checks format reward for thinking format and answer format with proper tags and structure.",
      scenario: "Structured reasoning output format validation",
      principles: null,
      module_path: "rm_gallery.gallery.rm.format.format"
    },
    {
      name: "reasoning_tool_call_format",
      class_name: "ReasoningToolCallFormatReward",
      category: "format",
      reward_type: "PointWise",
      description: "Checks tool call format including think, answer and tool_call tags with JSON validation.",
      scenario: "Tool-using agent response format validation",
      principles: null,
      module_path: "rm_gallery.gallery.rm.format.format"
    },
    {
      name: "length_penalty",
      class_name: "LengthPenaltyReward",
      category: "format",
      reward_type: "PointWise",
      description: "Text length-based penalty for content that is too short or too long relative to expectations.",
      scenario: "Response length control and optimization",
      principles: null,
      module_path: "rm_gallery.gallery.rm.format.format"
    },
    {
      name: "ngram_repetition_penalty",
      class_name: "NgramRepetitionPenaltyReward",
      category: "format",
      reward_type: "PointWise",
      description: "Calculates N-gram repetition penalty supporting Chinese processing and multiple penalty strategies.",
      scenario: "Repetitive content detection and penalization",
      principles: null,
      module_path: "rm_gallery.gallery.rm.format.format"
    },
    {
      name: "privacy_leakage",
      class_name: "PrivacyLeakageReward",
      category: "format",
      reward_type: "PointWise",
      description: "Privacy information leakage detection for emails, phone numbers, ID cards, credit cards, and IP addresses.",
      scenario: "Privacy protection and PII detection",
      principles: null,
      module_path: "rm_gallery.gallery.rm.format.format"
    }
  ];

  // —— Utils
  function show(el){ el.hidden = false; }
  function hide(el){ el.hidden = true; }
  function setLoading(on){
    on ? (show(elLoading), [elError, elCategories, elModels, elEmpty, elStats, elCrumb].forEach(hide))
       : hide(elLoading);
  }
  function setError(on){ on ? (show(elError), [elLoading].forEach(hide)) : hide(elError); }
  function clampTxt(s, n){ if(!s) return ""; return s.length<=n? s : s.slice(0,n)+"…"; }
  function debounce(fn, ms=250){ let t; return (...a)=>{ clearTimeout(t); t=setTimeout(()=>fn(...a), ms); }; }

  // —— Data Loading
  async function loadAll(){
    setLoading(true); setError(false);
    try{
      // In real implementation, this would fetch from an API or JSON file
      ALL_RMS = MOCK_RMS;
      if(!ALL_RMS.length) throw new Error("no data");

      GROUPED_RMS = ALL_RMS.reduce((acc, rm)=>{
        (acc[rm.category] ||= []).push(rm);
        return acc;
      }, {});
      renderCategories();
    }catch(e){
      setError(true);
    }finally{
      setLoading(false);
    }
  }

  // —— Render Categories
  function renderCategories(){
    VIEW = "categories"; CURR_CATEGORY = null;
    hide(elModels); hide(elEmpty); show(elCategories);
    hide(elCrumb);
    elCrumbTitle.textContent = "RM Categories";
    elType.textContent = "reward models";

    const availableCategories = Object.keys(GROUPED_RMS);

    const sections = Object.entries(CATEGORY_MAP).map(([categoryName, prefixes])=>{
      const categories = prefixes.filter(p => availableCategories.includes(p));
      const allRMs = categories.flatMap(cat => GROUPED_RMS[cat] || []);

      if (!allRMs.length) return "";

      const itemsHtml = allRMs.map((rm)=>{
        const rmIdx = GROUPED_RMS[rm.category].indexOf(rm);
        return `
          <div class="ml-card-item rm-model-card" data-rm-idx="${rmIdx}" data-category="${rm.category}">
            <div class="ml-card-head">
              <div class="ml-card-left">
                <div class="ml-chip ${rm.category}">${CATEGORY_CHIP_NAMES[rm.category] || rm.category.toUpperCase()}</div>
                <div class="ml-chip ${rm.reward_type === 'ListWise' ? 'success' : 'warning'}">${rm.reward_type.toUpperCase()}</div>
              </div>
            </div>
            <div class="ml-card-title-main">${rm.name}</div>
            <div class="ml-card-class">${rm.class_name}</div>
            <div class="ml-card-sample">${clampTxt(rm.description, 135)}</div>
            <div class="ml-card-foot">
              <span style="opacity: 0.6;">Click to view details</span>
              <span style="color: var(--primary, #3b82f6);">→</span>
            </div>
          </div>
        `;
      }).join("");

      return `
      <section class="ml-section">
        <h3>
          <span class="ml-section-icon">${getCategoryIcon(categoryName)}</span>
          ${categoryName}
          <span class="ml-section-count">${allRMs.length} models</span>
        </h3>
        <div class="ml-grid">
          ${itemsHtml}
        </div>
      </section>
      `;
    }).join("");

    elCategories.innerHTML = sections;
    bindModelClicks();

    show(elStats);
    const totalRMs = ALL_RMS.length;
    elCount.textContent = totalRMs;
    elTotal.textContent = totalRMs;
  }

  // Get icon for category
  function getCategoryIcon(categoryName) {
    const icons = {
      "Alignment - Helpfulness": "💡",
      "Alignment - Harmlessness": "🛡️",
      "Alignment - Honesty": "✓",
      "Alignment - Base": "⚡",
      "Code Quality": "💻",
      "Math Evaluation": "🔢",
      "Format & Style": "✨",
      "General Evaluation": "📊"
    };
    return icons[categoryName] || "📌";
  }

  // —— Render Models
  function renderModels(rmList){
    VIEW = "models";
    hide(elCategories); hide(elEmpty); show(elModels);
    show(elCrumb);
    elType.textContent = "reward models";
    elCrumbTitle.textContent = `Exploring ${CURR_CATEGORY}`;

    if(!rmList.length){
      hide(elModels); show(elEmpty); hide(elStats); return;
    }

    elModels.innerHTML = rmList.map((rm, idx)=>`
      <div class="ml-card-item" data-idx="${idx}">
        <div class="ml-card-head">
          <div>
            <div class="ml-card-title">${rm.name}</div>
            <div class="ml-card-sub">${rm.class_name}</div>
          </div>
          <div class="ml-chip ${rm.reward_type === 'ListWise' ? 'success' : 'warning'}">${rm.reward_type.toUpperCase()}</div>
        </div>
        <div class="ml-card-sample">${clampTxt(rm.description, 120)}</div>
        <div class="ml-card-foot">
          <span>🏷️ ${rm.category}</span>
          <span>Details →</span>
        </div>
      </div>
    `).join("");

    // Modal binding
    [...elModels.querySelectorAll(".ml-card-item")].forEach(card=>{
      card.addEventListener("click", ()=>{
        const idx = Number(card.getAttribute("data-idx"));
        const rm = rmList[idx];
        showRMModal(rm);
      });
    });

    show(elStats);
    elCount.textContent = rmList.length;
    elTotal.textContent = rmList.length;
  }

  function showRMModal(rm) {
    mCategory.textContent = rm.category;
    mCategory.className = `ml-chip ${rm.category}`;
    mType.textContent = rm.reward_type;
    mDescription.textContent = rm.description;
    mScenario.textContent = rm.scenario;

    // Handle principles
    if (rm.principles && rm.principles.length > 0) {
      const principlesList = rm.principles.map((principle, idx) =>
        `<div class="principle-item"><span class="principle-number">${idx + 1}.</span>${principle}</div>`
      ).join("");
      mPrinciples.innerHTML = `<div class="principle-list">${principlesList}</div>`;
      show(principlesSection);
    } else {
      hide(principlesSection);
    }

    // Usage example
    const usageExample = `from rm_gallery.core.reward.registry import RewardRegistry

# Initialize the reward model
rm = RewardRegistry.get("${rm.name}")

# Use the reward model
result = rm.evaluate(sample)
print(result)`;
    mUsage.textContent = usageExample;

    // Registry info
    mRegistry.textContent = rm.name;
    mClass.textContent = rm.class_name;
    mModule.textContent = rm.module_path;
    mRewardType.textContent = rm.reward_type;

    dlg.showModal();
  }

  function bindModelClicks(){
    [...elCategories.querySelectorAll(".rm-model-card")].forEach(card=>{
      card.addEventListener("click", ()=>{
        const category = card.getAttribute("data-category");
        const rmIdx = Number(card.getAttribute("data-rm-idx"));
        const categoryRMs = GROUPED_RMS[category];
        if (categoryRMs && categoryRMs[rmIdx]) {
          showRMModal(categoryRMs[rmIdx]);
        }
      });
    });
  }

  // —— Search
  function handleSearch(){
    const q = elSearch.value.trim().toLowerCase();
    if(!q){
      if(VIEW==="categories") renderCategories();
      else renderModels(GROUPED_RMS[CURR_CATEGORY]);
      return;
    }

    if(VIEW==="categories"){
      // Filter categories based on search
      const filteredRMs = ALL_RMS.filter(rm =>
        rm.name.toLowerCase().includes(q) ||
        rm.description.toLowerCase().includes(q) ||
        rm.category.toLowerCase().includes(q) ||
        rm.class_name.toLowerCase().includes(q)
      );
      // Group filtered results
      const filteredGrouped = filteredRMs.reduce((acc, rm)=>{
        (acc[rm.category] ||= []).push(rm);
        return acc;
      }, {});
      const backup = {...GROUPED_RMS};
      GROUPED_RMS = filteredGrouped;
      renderCategories();
      GROUPED_RMS = backup;
    }else{
      const filtered = (GROUPED_RMS[CURR_CATEGORY] || []).filter(rm =>
        rm.name.toLowerCase().includes(q) ||
        rm.description.toLowerCase().includes(q) ||
        rm.class_name.toLowerCase().includes(q)
      );
      renderModels(filtered);
    }
  }

  // —— Events
  function initEvents() {
    elRetry?.addEventListener("click", loadAll);
    elBack?.addEventListener("click", ()=> renderCategories());
    elSearch?.addEventListener("input", debounce(handleSearch, 250));
    elClear?.addEventListener("click", ()=>{
      elSearch.value = ""; handleSearch();
    });

    // Close modal when clicking outside
    dlg?.addEventListener("click", (e)=> {
      const rect = dlg.querySelector('.ml-modal-card')?.getBoundingClientRect();
      if (rect && (e.clientX < rect.left || e.clientX > rect.right ||
                   e.clientY < rect.top || e.clientY > rect.bottom)) {
        dlg.close();
      }
    });
  }

  // —— Init
  document.addEventListener("DOMContentLoaded", ()=> {
    initEvents();
    loadAll();
  });
})();
</script>