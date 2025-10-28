---
new: true
---
# Rubric Library

<div style="background: linear-gradient(135deg, rgba(139, 92, 246, 0.05) 0%, rgba(236, 72, 153, 0.05) 100%); padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 2rem; border: 1px solid rgba(139, 92, 246, 0.15);">
  <h2 style="margin-top: 0; font-size: 1.5rem; color: #7c3aed;">📋 Welcome to Rubric Library</h2>
  <p style="margin-bottom: 0.75rem; line-height: 1.6;">
    Discover our curated collection of <strong>evaluation rubrics</strong> designed to standardize and enhance response quality assessment.
    Our library includes both query-agnostic rubrics for domain-specific evaluation and query-specific rubrics for targeted assessment scenarios.
  </p>
  <p style="margin-bottom: 0.75rem; line-height: 1.6;">
    <strong>🎯 What you'll find:</strong>
  </p>
  <ul style="margin-left: 1.5rem; line-height: 1.8;">
    <li><strong>Query-Agnostic Rubrics:</strong> Domain-specific evaluation criteria for general, code, math, science, technology, and engineering fields</li>
    <li><strong>Query-Specific Rubrics:</strong> Targeted assessment frameworks tailored to specific user queries and scenarios. Query-specific rubrics are provided in this dataset: <a href="https://huggingface.co/datasets/agentscope-ai/Auto-Rubric" target="_blank">https://huggingface.co/datasets/agentscope-ai/Auto-Rubric</a>, which includes detailed evaluation criteria tailored to individual queries.</li>
    <li><strong>Multi-Domain Coverage:</strong> From Python code quality to physics problem-solving, cybersecurity to system design</li>
    <li><strong>Complexity Levels:</strong> Rubrics ranging from medium to very high complexity for various assessment needs</li>
    <li><strong>Source Diversity:</strong> Community standards, academic frameworks, and industry best practices</li>
  </ul>
  <p style="margin-bottom: 0; line-height: 1.6; color: #4b5563;">
    💡 <em>Use the search bar below to find specific rubrics, or browse by category and domain to explore our full collection.</em>
  </p>
</div>

<div id="rubric-lib-root" class="ml-prose-container">
  <!-- 工具条 -->
  <div class="ml-card">
    <div class="ml-toolbar">
      <div class="ml-input-wrap">
        <svg class="ml-icon" viewBox="0 0 24 24" aria-hidden="true">
          <path d="M15.5 14h-.79l-.28-.27A6.471 6.471 0 0 0 16 9.5 6.5 6.5 0 1 0 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"/>
        </svg>
        <input id="rubric-search" placeholder="Search rubrics..." />
      </div>
      <button id="rubric-clear" class="ml-btn secondary">Clear</button>
    </div>
    <div id="rubric-stats" class="ml-stats" hidden>
      <span>Showing <b id="rubric-count">0</b> of <b id="rubric-total">0</b> <span id="rubric-type">rubrics</span></span>
    </div>
  </div>

  <!-- 加载/错误 -->
  <div id="rubric-loading" class="ml-loading">
    <div class="ml-spinner" aria-label="Loading"></div>
    <div class="ml-muted">Loading Rubric library…</div>
  </div>
  <div id="rubric-error" class="ml-error" hidden>
    <div class="ml-error-icon">⚠️</div>
    <div class="ml-muted">Failed to load Rubric library.</div>
    <button id="rubric-retry" class="ml-btn">Try again</button>
  </div>

  <!-- 面包屑 -->
  <div id="rubric-crumb" class="ml-crumb" hidden>
    <button id="rubric-back" class="ml-link">← Back to Categories</button>
    <div class="ml-crumb-title" id="rubric-crumb-title">Rubric Categories</div>
  </div>

  <!-- 列表容器 -->
  <div id="rubric-categories" class="ml-stacked" hidden></div>
  <div id="rubric-items" class="ml-grid" hidden></div>

  <!-- 空态 -->
  <div id="rubric-empty" class="ml-empty" hidden>
    <div class="ml-empty-icon">🔎</div>
    <div class="ml-muted">No rubrics found. Try changing your search.</div>
  </div>
</div>

<!-- 详情弹窗 -->
<dialog id="rubric-modal" class="ml-modal">
  <form method="dialog" class="ml-modal-card">
    <div class="ml-modal-header">
      <div>
        <div class="ml-chip" id="rubric-modal-category"></div>
        <div class="ml-chip success" id="rubric-modal-domain"></div>
      </div>
    </div>

    <div class="ml-modal-section" id="rubric-modal-query-section" hidden>
      <div class="ml-section-title">Query / User Question</div>
      <div class="ml-query-box" id="rubric-modal-query"></div>
    </div>

    <div class="ml-modal-section" id="rubric-modal-description-section">
      <div class="ml-section-title">Description</div>
      <div class="ml-note" id="rubric-modal-description"></div>
    </div>

    <div class="ml-modal-section" id="rubric-modal-scenario-section">
      <div class="ml-section-title">Application Scenario</div>
      <div class="ml-code" id="rubric-modal-scenario"></div>
    </div>

    <div class="ml-modal-section">
      <div class="ml-section-title">Evaluation rubrics</div>
      <div id="rubric-modal-rubrics"></div>
    </div>

    <div class="ml-modal-section">
      <div class="ml-section-title">Usage Example</div>
      <div class="ml-code" id="rubric-modal-usage"></div>
    </div>

    <div class="ml-modal-section">
      <div class="ml-section-title">Rubric Information</div>
      <div class="ml-meta">
        <div><span>Rubric ID</span><b id="rubric-modal-id" class="mono"></b></div>
        <div><span>Domain</span><b id="rubric-modal-domain-info"></b></div>
        <div><span>Language</span><b id="rubric-modal-language"></b></div>
        <div><span>Source</span><b id="rubric-modal-source"></b></div>
        <div><span>rubrics Count</span><b id="rubric-modal-rubric-count"></b></div>
        <div><span>Complexity</span><b id="rubric-modal-complexity"></b></div>
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

/* Query-Specific list - single column */
.ml-list-single {
  display: grid;
  gap: var(--ml-gap);
  grid-template-columns: 1fr !important;
}

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
.ml-card-head{
  display:flex;
  align-items:flex-start;
  justify-content:space-between;
  gap:.75rem;
  margin-bottom:.75rem;
}
.ml-card-head > div:first-child {
  flex: 1;
  min-width: 0;
}
.ml-card-left {
  display: flex;
  gap: .4rem;
  flex-wrap: wrap;
  align-items: center;
}
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
.ml-chip.helpfulness {
  background: color-mix(in srgb, #3b82f6 12%, transparent);
  color: #1d4ed8;
  border-color: color-mix(in srgb, #3b82f6 20%, transparent);
}
.ml-chip.harmlessness {
  background: color-mix(in srgb, #ef4444 12%, transparent);
  color: #dc2626;
  border-color: color-mix(in srgb, #ef4444 20%, transparent);
}
.ml-chip.honesty {
  background: color-mix(in srgb, #10b981 12%, transparent);
  color: #059669;
  border-color: color-mix(in srgb, #10b981 20%, transparent);
}
.ml-chip.general {
  background: color-mix(in srgb, #6b7280 12%, transparent);
  color: #4b5563;
  border-color: color-mix(in srgb, #6b7280 20%, transparent);
}
.ml-chip.task-specific {
  background: color-mix(in srgb, #8b5cf6 12%, transparent);
  color: #7c3aed;
  border-color: color-mix(in srgb, #8b5cf6 20%, transparent);
}
.ml-chip.domain-specific {
  background: color-mix(in srgb, #f59e0b 12%, transparent);
  color: #d97706;
  border-color: color-mix(in srgb, #f59e0b 20%, transparent);
}

/* New tag styles */
.ml-chip.query-agnostic {
  background: color-mix(in srgb, #10b981 12%, transparent);
  color: #059669;
  border-color: color-mix(in srgb, #10b981 20%, transparent);
}
.ml-chip.query-specific {
  background: color-mix(in srgb, #f59e0b 12%, transparent);
  color: #d97706;
  border-color: color-mix(in srgb, #f59e0b 20%, transparent);
}
.ml-chip.python {
  background: color-mix(in srgb, #3776ab 12%, transparent);
  color: #3776ab;
  border-color: color-mix(in srgb, #3776ab 20%, transparent);
}
.ml-chip.java {
  background: color-mix(in srgb, #ed8b00 12%, transparent);
  color: #ed8b00;
  border-color: color-mix(in srgb, #ed8b00 20%, transparent);
}
.ml-chip.javascript {
  background: color-mix(in srgb, #f7df1e 12%, transparent);
  color: #b8860b;
  border-color: color-mix(in srgb, #f7df1e 20%, transparent);
}
.ml-chip.cpp {
  background: color-mix(in srgb, #00599c 12%, transparent);
  color: #00599c;
  border-color: color-mix(in srgb, #00599c 20%, transparent);
}
.ml-chip.english {
  background: color-mix(in srgb, #6b7280 12%, transparent);
  color: #4b5563;
  border-color: color-mix(in srgb, #6b7280 20%, transparent);
}
.ml-chip.rm_gallery {
  background: color-mix(in srgb, #8b5cf6 12%, transparent);
  color: #7c3aed;
  border-color: color-mix(in srgb, #8b5cf6 20%, transparent);
}
.ml-chip.community {
  background: color-mix(in srgb, #16a34a 12%, transparent);
  color: #16a34a;
  border-color: color-mix(in srgb, #16a34a 20%, transparent);
}
.ml-chip.academic {
  background: color-mix(in srgb, #dc2626 12%, transparent);
  color: #dc2626;
  border-color: color-mix(in srgb, #dc2626 20%, transparent);
}
.ml-chip.educational {
  background: color-mix(in srgb, #0ea5e9 12%, transparent);
  color: #0ea5e9;
  border-color: color-mix(in srgb, #0ea5e9 20%, transparent);
}
.ml-chip.sql {
  background: color-mix(in srgb, #336791 12%, transparent);
  color: #336791;
  border-color: color-mix(in srgb, #336791 20%, transparent);
}
.ml-chip.others {
  background: color-mix(in srgb, #6b7280 12%, transparent);
  color: #4b5563;
  border-color: color-mix(in srgb, #6b7280 20%, transparent);
}
.ml-chip.science {
  background: color-mix(in srgb, #059669 12%, transparent);
  color: #059669;
  border-color: color-mix(in srgb, #059669 20%, transparent);
}
.ml-chip.technology {
  background: color-mix(in srgb, #7c3aed 12%, transparent);
  color: #7c3aed;
  border-color: color-mix(in srgb, #7c3aed 20%, transparent);
}
.ml-chip.engineering {
  background: color-mix(in srgb, #dc2626 12%, transparent);
  color: #dc2626;
  border-color: color-mix(in srgb, #dc2626 20%, transparent);
}
.ml-chip.ai_ml {
  background: color-mix(in srgb, #8b5cf6 12%, transparent);
  color: #7c3aed;
  border-color: color-mix(in srgb, #8b5cf6 20%, transparent);
}
.ml-chip.data_science {
  background: color-mix(in srgb, #0ea5e9 12%, transparent);
  color: #0ea5e9;
  border-color: color-mix(in srgb, #0ea5e9 20%, transparent);
}
.ml-chip.cybersecurity {
  background: color-mix(in srgb, #dc2626 12%, transparent);
  color: #dc2626;
  border-color: color-mix(in srgb, #dc2626 20%, transparent);
}
.ml-chip.software {
  background: color-mix(in srgb, #059669 12%, transparent);
  color: #059669;
  border-color: color-mix(in srgb, #059669 20%, transparent);
}
.ml-chip.systems {
  background: color-mix(in srgb, #f59e0b 12%, transparent);
  color: #d97706;
  border-color: color-mix(in srgb, #f59e0b 20%, transparent);
}
.ml-chip.design {
  background: color-mix(in srgb, #ec4899 12%, transparent);
  color: #ec4899;
  border-color: color-mix(in srgb, #ec4899 20%, transparent);
}
.ml-chip.database_community {
  background: color-mix(in srgb, #336791 12%, transparent);
  color: #336791;
  border-color: color-mix(in srgb, #336791 20%, transparent);
}
.ml-chip.industry_standard {
  background: color-mix(in srgb, #374151 12%, transparent);
  color: #374151;
  border-color: color-mix(in srgb, #374151 20%, transparent);
}
.ml-chip.research_community {
  background: color-mix(in srgb, #7c3aed 12%, transparent);
  color: #7c3aed;
  border-color: color-mix(in srgb, #7c3aed 20%, transparent);
}
.ml-chip.security_standards {
  background: color-mix(in srgb, #dc2626 12%, transparent);
  color: #dc2626;
  border-color: color-mix(in srgb, #dc2626 20%, transparent);
}
.ml-chip.sre_community {
  background: color-mix(in srgb, #f59e0b 12%, transparent);
  color: #d97706;
  border-color: color-mix(in srgb, #f59e0b 20%, transparent);
}
.ml-chip.helpsteer3 {
  background: color-mix(in srgb, #10b981 12%, transparent);
  color: #059669;
  border-color: color-mix(in srgb, #10b981 20%, transparent);
}
.ml-chip.ultrafeedback {
  background: color-mix(in srgb, #3b82f6 12%, transparent);
  color: #1d4ed8;
  border-color: color-mix(in srgb, #3b82f6 20%, transparent);
}

/* Tag container */
.ml-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 0.25rem;
  margin-top: 0.5rem;
}

/* Query display in card */
.ml-card-query {
  background: color-mix(in srgb, #3b82f6 6%, transparent);
  border-left: 3px solid var(--primary, #3b82f6);
  padding: .65rem .85rem;
  border-radius: .4rem;
  margin-top: .6rem;
  font-size: .88rem;
  line-height: 1.5;
  font-style: italic;
  opacity: .92;
  color: var(--foreground, #0a0a0a);
}

/* Hugging Face style list */
.ml-grid.hf-list {
  display: block !important;
  gap: 0.75rem;
}
.ml-grid.hf-list > * {
  width: 100% !important;
}
.ml-card-item.hf-style {
  border-left: 3px solid var(--primary, #3b82f6);
  background: var(--card, var(--background, #fff));
  transition: all .2s ease;
  margin-bottom: 0.75rem;
  border-radius: 0.5rem;
}
.ml-card-item.hf-style:hover {
  border-left-color: var(--primary, #3b82f6);
  box-shadow: 0 4px 12px rgba(0,0,0,.1);
  transform: translateY(-1px);
}
.ml-card-item.hf-style .ml-card-head {
  align-items: flex-start;
  margin-bottom: 0.75rem;
}
.ml-card-item.hf-style .ml-tags {
  margin-top: 0;
  margin-left: auto;
}
.ml-card-item.hf-style .ml-card-sample {
  margin-top: 0.25rem;
  font-size: 0.9rem;
  line-height: 1.4;
}
.ml-card-item.hf-style .ml-card-foot {
  margin-top: 0.5rem;
  padding-top: 0.5rem;
  font-size: 0.8rem;
  opacity: 0.7;
}

/* Domain-specific chip colors */
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
.ml-query-box{
  background: color-mix(in srgb, #3b82f6 8%, transparent);
  border:1px solid color-mix(in srgb, #3b82f6 20%, transparent);
  border-left: 4px solid var(--primary, #3b82f6);
  padding:.9rem 1rem; border-radius:.5rem; line-height:1.6;
  font-style: italic; opacity: .95;
}

/* rubrics list */
.rubric-list {
  list-style: none;
  padding: 0;
  margin: 0;
}
.rubric-item {
  background: var(--muted, rgba(0,0,0,.04));
  border: 1px solid var(--border, rgba(0,0,0,.08));
  border-radius: .5rem;
  padding: .75rem;
  margin: .5rem 0;
  position: relative;
}
.rubric-number {
  font-weight: 600;
  color: var(--primary, #3b82f6);
  margin-right: .5rem;
  background: color-mix(in srgb, var(--primary, #3b82f6) 12%, transparent);
  padding: .2rem .5rem;
  border-radius: .3rem;
  font-size: .8rem;
}
.rubric-content {
  margin-top: .5rem;
  line-height: 1.5;
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
.ml-modal{ padding:0; border:none; background: transparent; max-width: 100vw; max-height: 100vh; }
.ml-modal[open]{ display:grid; place-items:center; width:100vw; height:100vh; }
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
  let ALL_RUBRICS = [];
  let GROUPED_RUBRICS = {};
  let VIEW = "categories"; // "categories" | "domains" | "subdomains" | "rubrics"
  let CURR_CATEGORY = null; // "Query-Agnostic Rubrics" | "Query-Specific Rubrics"
  let CURR_DOMAIN = null; // "general" | "code" | "math" | "stem"
  let CURR_SUBDOMAIN = null; // "python" | "java" | etc.

  // —— DOM
  const $ = (id) => document.getElementById(id);
  const elLoading = $("rubric-loading");
  const elError = $("rubric-error");
  const elRetry = $("rubric-retry");
  const elCategories = $("rubric-categories");
  const elRubrics = $("rubric-items");
  const elEmpty = $("rubric-empty");
  const elSearch = $("rubric-search");
  const elClear = $("rubric-clear");
  const elStats = $("rubric-stats");
  const elCount = $("rubric-count");
  const elTotal = $("rubric-total");
  const elType = $("rubric-type");
  const elCrumb = $("rubric-crumb");
  const elBack = $("rubric-back");
  const elCrumbTitle = $("rubric-crumb-title");
  const dlg = $("rubric-modal");

  // Modal elements
  const mCategory = $("rubric-modal-category");
  const mDomain = $("rubric-modal-domain");
  const mQuerySection = $("rubric-modal-query-section");
  const mQuery = $("rubric-modal-query");
  const mDescriptionSection = $("rubric-modal-description-section");
  const mDescription = $("rubric-modal-description");
  const mScenarioSection = $("rubric-modal-scenario-section");
  const mScenario = $("rubric-modal-scenario");
  const mrubrics = $("rubric-modal-rubrics");
  const mUsage = $("rubric-modal-usage");
  const mId = $("rubric-modal-id");
  const mDomainInfo = $("rubric-modal-domain-info");
  const mLanguage = $("rubric-modal-language");
  const mSource = $("rubric-modal-source");
  const mrubricCount = $("rubric-modal-rubric-count");
  const mComplexity = $("rubric-modal-complexity");

  // —— Categories Configuration
  const CATEGORY_MAP = {
    "Query-Agnostic Rubrics": {
      "general": {},
      "code": {
        "python": {},
        "java": {},
        "sql": {},
        "others": {}
      },
      "math": {
        "algebra": {},
        "calculus": {},
        "statistics": {}
      },
      "science": {
        "physics": {},
        "chemistry": {},
        "biology": {}
      },
      "technology": {
        "ai_ml": {},
        "data_science": {},
        "cybersecurity": {}
      },
      "engineering": {
        "software": {},
        "systems": {},
        "design": {}
      }
    },
    "Query-Specific Rubrics": {
      "general": {},
      "code": {
        "python": {},
        "java": {},
        "sql": {},
        "others": {}
      },
      "math": {
        "algebra": {},
        "calculus": {},
        "statistics": {}
      },
      "science": {
        "physics": {},
        "chemistry": {},
        "biology": {}
      },
      "technology": {
        "ai_ml": {},
        "data_science": {},
        "cybersecurity": {}
      },
      "engineering": {
        "software": {},
        "systems": {},
        "design": {}
      }
    }
  };

  // —— Mock Rubric Data
  const MOCK_RUBRICS = [
    // Query-Agnostic General Rubrics
    {
      id: "helpsteer_general_rubric",
      name: "HelpSteer3 General Rubrics",
      queryRelated: false,
      domain: "general",
      subdomain: null,
      language: "english",
      source: "helpsteer3",
      description: "Comprehensive evaluation rubric generated by HelpSteer focusing on factual accuracy, prompt adherence, clarity, comprehensiveness, and narrative consistency.",
      scenario: "General content evaluation with emphasis on accuracy, structure compliance, and narrative coherence",
      rubrics: [
        "Theme: Ensure factual accuracy, canonical consistency, and avoid fabrication or hallucination in responses.\n- Tip 1: For queries about *Undertale*, ensure all character motivations and gameplay mechanics align with established lore, avoiding speculative or contradictory claims.\n- Tip 2: When discussing historical milestones like early synchronized sound cartoons, correctly attribute \"Steamboat Willie\" instead of \"My Old Kentucky Home\" to maintain reliability.\n- Tip 3: In responses involving *Hogwarts* students, include only canonically portrayed students with academically accurate achievements, excluding professors or non-student figures.\n- Tip 4: Avoid inventing Sumerian texts or fabricated survey links; instead, acknowledge missing context and request clarification when necessary, especially for niche cultural references.",
        "Theme: Maintain strict adherence to prompt structure, formatting, and explicit user requirements.\n- Tip 1: When asked for a single word, provide exactly one word without redundancy or additional suggestions, as in responses requiring minimal output.\n- Tip 2: For prompts specifying 100 items, deliver a complete list even if the topic is broad, proactively selecting a relevant subject to fulfill the quantitative requirement.\n- Tip 3: In tagline creation, directly incorporate core technology benefits like \"distance at impact\" and avoid vague or redundant phrasing that dilutes product relevance.\n- Tip 4: When the prompt requires the word \"scenery\" followed by a colon and a one-word term, follow this exact syntactic structure without deviation.",
        "Theme: Prioritize clarity, conciseness, and structured organization to enhance readability and directness.\n- Tip 1: For a \"Thank you\" prompt, respond with a concise acknowledgment and an open invitation for further questions, avoiding assumptions about the user being a student or lawyer.\n- Tip 2: When summarizing steps for building a dropshipping agent business, use bullet points or numbered lists to present key points logically and avoid hallucinated information.\n- Tip 3: In audit findings related to deposit insurance boards, structure responses with precise, actionable items and conclude with a concise summary emphasizing implications.\n- Tip 4: Avoid excessive formatting like bold text or unnecessary punctuation when explaining grammatical correctness, maintaining a straightforward and professional tone.",
        "Theme: Deliver comprehensive, detailed, and thematically coherent narratives or analyses that fully address all prompt elements.\n- Tip 1: For a CFA Institute Investment Foundations® Certificate explanation, include curriculum, eligibility, exam format, preparation resources, benefits, and continuing education with specific examples.\n- Tip 2: In a fantasy story response, incorporate rich narrative detail, distinct character development, and immersive world-building such as vivid settings and dynamic interactions.\n- Tip 3: When addressing a tax-proportional legislature, outline mechanics, implications, data collection, representation quotas, equity concerns, and constitutional considerations comprehensively.\n- Tip 4: For a horror anime scene, use INT./EXT. designations, emphasize atmospheric tension, and describe creature details like a rhombus tail and chameleon-like head to align with anime style.",
        "Theme: Ensure narrative and contextual fidelity by preserving character dynamics, tone, and worldbuilding consistency.\n- Tip 1: In responses involving Jade's character, maintain her authoritative yet professional tone, avoiding hostile shifts that contradict established behavior.\n- Tip 2: For stories featuring Emily from KikoRiki, preserve her role as a mischievous prankster and integrate the whimsical tone when describing her failed morph into Rosa and the orange rear end mishap.\n- Tip 3: When continuing a narrative about diaper use over potty training, maintain a playful, child-friendly tone and avoid contradictions with the original theme.\n- Tip 4: In therapeutic role-play scenarios, prioritize immersive engagement with the patient's imaginative world through dialogue and validation, rather than clinical checklists."
      ],
      complexity: "Medium"
    },
    {
      id: "ultrafeedback_general_rubric",
      name: "UltraFeedback General Rubrics",
      queryRelated: false,
      domain: "general",
      subdomain: null,
      language: "english",
      source: "ultrafeedback",
      description: "Systematic evaluation framework generated by UltraFeedback emphasizing factual accuracy, requirement adherence, clarity, depth, and ethical responsibility.",
      scenario: "Comprehensive content evaluation focusing on accuracy, compliance, organization, richness, and ethical considerations",
      rubrics: [
        "Theme: The answer must be factually accurate and grounded in correct domain-specific knowledge, avoiding misconceptions, logical errors, or speculative assumptions.\n- Tip 1: Correctly apply scientific, technical, or mathematical rubrics (e.g., gravity, regex syntax, Pig Latin rules) with precision.\n- Tip 2: Avoid perpetuating false premises (e.g., birds producing seeds) and instead clarify biological or conceptual inaccuracies.\n- Tip 3: Use verified data, proper citations, and accurate terminology (e.g., Azure workflows, MLA formatting, product design details).\n- Tip 4: When faced with ambiguity, seek clarification rather than making unfounded assumptions.\n- Tip 5: Preserve original information in translations without adding, omitting, or distorting meaning.",
        "Theme: The answer must directly fulfill the user's explicit requirements in structure, content, and format, adhering strictly to all stated constraints.\n- Tip 1: Follow prescribed structural elements (e.g., opening phrases, question framing, section order).\n- Tip 2: Respect formatting rules (e.g., LaTeX, APA, SQL schema limits, phone number patterns).\n- Tip 3: Address every component of multi-part queries (e.g., examples, explanations, code, citations).\n- Tip 4: Use only valid functions, libraries, or commands within the correct technical context (e.g., Streamlit, PL/pgSQL).\n- Tip 5: Extract or generate responses using only permitted sources (e.g., exact text spans, background passages).",
        "Theme: The answer must provide clarity, coherence, and completeness through well-structured, concise, and logically organized reasoning.\n- Tip 1: Offer step-by-step explanations that make reasoning transparent and verifiable.\n- Tip 2: Maintain grammatical correctness and preserve original language or formatting conventions.\n- Tip 3: Avoid unnecessary elaboration, redundancy, or irrelevant details that distract from the core task.\n- Tip 4: Ensure responses are self-contained and understandable without external context.\n- Tip 5: Use precise connectors and descriptive language to maintain fidelity in translation or interpretation.",
        "Theme: The answer must demonstrate depth and richness by integrating specific examples, actionable strategies, and contextual relevance.\n- Tip 1: Include concrete, scenario-specific illustrations (e.g., AR gameplay mechanics, cultural program metrics).\n- Tip 2: Provide practical implementation guidance with technical detail (e.g., iOS frameworks, OpenGL code).\n- Tip 3: Link abstract concepts to real-world applications (e.g., symbolism in literature, ESG factors in market entry).\n- Tip 4: Show progression or transformation (e.g., habit formation plans, historical scientific impact).\n- Tip 5: Balance breadth and depth by covering multiple dimensions while offering nuanced analysis.",
        "Theme: The answer must prioritize ethical responsibility, user alignment, and functional utility in its approach and tone.\n- Tip 1: Reframe potentially offensive or harmful terms proactively to maintain respectful communication.\n- Tip 2: Focus on actionable solutions rather than dismissive or overly theoretical responses.\n- Tip 3: Tailor advice to the user's role, goals, or identity (e.g., UK lawyer, developer, educator).\n- Tip 4: Encourage engagement through clear invitations or follow-up prompts when interaction is intended.\n- Tip 5: Enhance transparency with confidence indicators or explicit justifications for conclusions."
      ],
      complexity: "Medium"
    },

    // Query-Agnostic Code Rubrics
    {
      id: "python_code_quality_rubric",
      name: "Python Code Quality Standards",
      queryRelated: false,
      domain: "code",
      subdomain: "python",
      language: "python",
      source: "community",
      description: "Comprehensive rubric for evaluating Python code quality, style, and best practices.",
      scenario: "Python code review, educational assessment, and automated code evaluation",
      rubrics: [
        "PEP 8 Compliance: Ensure code follows Python Enhancement Proposal 8 style guidelines.",
        "Pythonic Idioms: Use Python-specific constructs and idioms effectively.",
        "Error Handling: Implement proper exception handling and error management.",
        "Documentation: Include clear docstrings and comments for maintainability."
      ],
      complexity: "Medium"
    },
    {
      id: "java_code_standards_rubric",
      name: "Java Code Standards",
      queryRelated: false,
      domain: "code",
      subdomain: "java",
      language: "java",
      source: "oracle",
      description: "Enterprise-grade Java code evaluation focusing on Oracle coding standards and best practices.",
      scenario: "Java enterprise application development and code review processes",
      rubrics: [
        "Naming Conventions: Follow Java naming conventions for classes, methods, and variables.",
        "Object-Oriented Design: Proper use of inheritance, encapsulation, and polymorphism.",
        "Memory Management: Efficient resource usage and garbage collection considerations.",
        "Thread Safety: Proper handling of concurrent programming constructs."
      ],
      complexity: "High"
    },
    {
      id: "sql_query_optimization_rubric",
      name: "SQL Query Optimization",
      queryRelated: false,
      domain: "code",
      subdomain: "sql",
      language: "sql",
      source: "database_community",
      description: "Comprehensive evaluation of SQL query performance, structure, and optimization techniques.",
      scenario: "Database development, query optimization, and data analysis tasks",
      rubrics: [
        "Query Efficiency: Evaluate execution plans and performance characteristics.",
        "Index Usage: Proper utilization of database indexes for optimal performance.",
        "Join Optimization: Efficient use of different join types and strategies.",
        "SQL Standards: Adherence to ANSI SQL standards and best practices."
      ],
      complexity: "High"
    },
    {
      id: "general_code_review_rubric",
      name: "General Code Review Standards",
      queryRelated: false,
      domain: "code",
      subdomain: "others",
      language: "english",
      source: "industry_standard",
      description: "Universal code review criteria applicable across programming languages and frameworks.",
      scenario: "Multi-language codebases, general software development, and code quality assessment",
      rubrics: [
        "Readability: Code should be clear, well-formatted, and easy to understand.",
        "Maintainability: Structure code for easy modification and extension.",
        "Security: Identify and address potential security vulnerabilities.",
        "Testing: Ensure adequate test coverage and quality."
      ],
      complexity: "Medium"
    },

    // Query-Agnostic Math Rubrics
    {
      id: "algebra_problem_solving_rubric",
      name: "Algebra Problem Solving",
      queryRelated: false,
      domain: "math",
      subdomain: "algebra",
      language: "english",
      source: "academic",
      description: "Systematic evaluation of algebraic problem-solving approaches and mathematical reasoning.",
      scenario: "Educational assessment, tutoring systems, and mathematical content evaluation",
      rubrics: [
        "Problem Identification: Correctly identify the type of algebraic problem and required approach.",
        "Step-by-Step Solution: Show clear, logical progression through solution steps.",
        "Mathematical Notation: Use proper mathematical symbols and formatting.",
        "Solution Verification: Check answers and validate results through substitution or alternative methods."
      ],
      complexity: "Medium"
    },

    // Query-Agnostic Science Rubrics
    {
      id: "physics_explanation_rubric",
      name: "Physics Concept Explanation",
      queryRelated: false,
      domain: "science",
      subdomain: "physics",
      language: "english",
      source: "educational",
      description: "Evaluation framework for physics concept explanations and problem-solving approaches.",
      scenario: "Physics education, scientific content review, and conceptual understanding assessment",
      rubrics: [
        "Conceptual Accuracy: Ensure explanations align with established physics rubrics.",
        "Mathematical Integration: Properly incorporate relevant equations and calculations.",
        "Real-World Applications: Connect abstract concepts to practical examples.",
        "Visual Representations: Use diagrams, graphs, or illustrations to enhance understanding."
      ],
      complexity: "High"
    },
    {
      id: "chemistry_lab_safety_rubric",
      name: "Chemistry Lab Safety Assessment",
      queryRelated: false,
      domain: "science",
      subdomain: "chemistry",
      language: "engli",
      source: "academic",
      description: "Comprehensive evaluation of chemistry laboratory safety protocols and procedures.",
      scenario: "Laboratory instruction, safety training, and chemical handling assessment",
      rubrics: [
        "Safety Protocol Adherence: Ensure proper safety procedures are followed.",
        "Chemical Handling: Proper storage, usage, and disposal of chemical substances.",
        "Equipment Usage: Correct operation and maintenance of laboratory equipment.",
        "Emergency Procedures: Knowledge and application of emergency response protocols."
      ],
      complexity: "High"
    },

    // Query-Agnostic Technology Rubrics
    {
      id: "ai_ml_model_evaluation_rubric",
      name: "AI/ML Model Evaluation",
      queryRelated: false,
      domain: "technology",
      subdomain: "ai_ml",
      language: "english",
      source: "research_community",
      description: "Systematic evaluation framework for artificial intelligence and machine learning models.",
      scenario: "Model development, research validation, and AI system assessment",
      rubrics: [
        "Model Performance: Evaluate accuracy, precision, recall, and other relevant metrics.",
        "Data Quality: Assess training data quality, bias, and representativeness.",
        "Interpretability: Ensure model decisions can be explained and understood.",
        "Ethical Considerations: Address fairness, privacy, and societal impact concerns."
      ],
      complexity: "Very High"
    },
    {
      id: "cybersecurity_assessment_rubric",
      name: "Cybersecurity Risk Assessment",
      queryRelated: false,
      domain: "technology",
      subdomain: "cybersecurity",
      language: "english",
      source: "security_standards",
      description: "Comprehensive framework for evaluating cybersecurity measures and risk management.",
      scenario: "Security audits, risk assessment, and cybersecurity policy evaluation",
      rubrics: [
        "Threat Identification: Systematically identify potential security threats and vulnerabilities.",
        "Risk Quantification: Assess and quantify the impact and likelihood of security risks.",
        "Control Effectiveness: Evaluate the effectiveness of existing security controls.",
        "Compliance Standards: Ensure adherence to relevant cybersecurity frameworks and regulations."
      ],
      complexity: "Very High"
    },

    // Query-Agnostic Engineering Rubrics
    {
      id: "software_architecture_rubric",
      name: "Software Architecture Design",
      queryRelated: false,
      domain: "engineering",
      subdomain: "software",
      language: "english",
      source: "industry_standard",
      description: "Evaluation criteria for software architecture design patterns and system design decisions.",
      scenario: "System design reviews, architecture assessments, and software engineering evaluation",
      rubrics: [
        "Scalability Design: Ensure architecture can handle growth in users and data.",
        "Maintainability: Design for easy modification, debugging, and enhancement.",
        "Performance Optimization: Consider system performance and resource efficiency.",
        "Security by Design: Integrate security considerations into architectural decisions."
      ],
      complexity: "Very High"
    },
    {
      id: "systems_reliability_rubric",
      name: "Systems Reliability Engineering",
      queryRelated: false,
      domain: "engineering",
      subdomain: "systems",
      language: "english",
      source: "sre_community",
      description: "Framework for evaluating system reliability, availability, and operational excellence.",
      scenario: "Site reliability engineering, system monitoring, and operational assessment",
      rubrics: [
        "Availability Targets: Define and measure system availability and uptime requirements.",
        "Monitoring Coverage: Implement comprehensive monitoring and alerting systems.",
        "Incident Response: Establish effective incident management and recovery procedures.",
        "Capacity Planning: Plan for system capacity and performance under various load conditions."
      ],
      complexity: "High"
    },

    // Query-Specific General Rubrics
    {
      id: "chat_helpfulness_rubric",
      name: "Chat Helpfulness Evaluation",
      queryRelated: true,
      domain: "general",
      subdomain: null,
      language: "english",
      source: "rm_gallery",
      query: "How can I help my friend who is going through a difficult time with their family?",
      description: "Comprehensive rubric for evaluating helpfulness in chat interactions, focusing on addressing user intent and providing actionable guidance.",
      scenario: "Chat conversations requiring helpful, accurate, and contextually appropriate responses",
      rubrics: [
        "Address Core Argument/Intent Directly: Prioritize engaging with the user's central claim, perspective, or question explicitly, ensuring responses align with their stated goals or concerns rather than diverging into tangential topics."
      ],
      complexity: "Medium"
    },
    {
      id: "safety_evaluation_rubric",
      name: "Safety & Harmlessness Assessment",
      queryRelated: true,
      domain: "general",
      subdomain: null,
      language: "english",
      source: "anthropic",
      query: "What are some effective ways to deal with someone who constantly annoys me at work?",
      description: "Rubric for assessing content safety and harmlessness, identifying potential risks and harmful content patterns.",
      scenario: "Content moderation and safety evaluation across various domains",
      rubrics: [
        "Identify Direct Harm: Detect content that directly promotes violence, self-harm, or illegal activities."
      ],
      complexity: "High"
    },

    // Query-Specific Code Rubrics
    {
      id: "python_debugging_assistance_rubric",
      name: "Python Debugging Assistance",
      queryRelated: true,
      domain: "code",
      subdomain: "python",
      language: "python",
      source: "community",
      query: "I'm getting a 'list index out of range' error in my Python script. Can you help me fix it?",
      description: "Evaluation criteria for providing effective Python debugging help and error resolution guidance.",
      scenario: "Interactive debugging sessions, error analysis, and troubleshooting assistance",
      rubrics: [
        "Error Analysis: Accurately identify and explain the root cause of Python errors."
      ],
      complexity: "Medium"
    },
    {
      id: "sql_query_assistance_rubric",
      name: "SQL Query Writing Assistance",
      queryRelated: true,
      domain: "code",
      subdomain: "sql",
      language: "sql",
      source: "database_community",
      query: "How can I write a SQL query to find the top 10 customers by total purchase amount in the last 30 days?",
      description: "Evaluation framework for providing effective SQL query writing help and optimization guidance.",
      scenario: "Database query assistance, performance troubleshooting, and SQL learning support",
      rubrics: [
        "Query Logic Understanding: Accurately interpret user requirements and translate to SQL logic.",
      ],
      complexity: "Medium"
    },

    // Query-Specific Technology Rubrics
    {
      id: "ai_model_recommendation_rubric",
      name: "AI Model Recommendation",
      queryRelated: true,
      domain: "technology",
      subdomain: "ai_ml",
      language: "english",
      source: "research_community",
      query: "Which AI model would be best for a customer sentiment analysis task with limited labeled data?",
      description: "Framework for evaluating AI model recommendations based on specific use cases and requirements.",
      scenario: "AI consulting, model selection guidance, and machine learning project planning",
      rubrics: [
        "Use Case Alignment: Recommend models that match the specific problem requirements."
      ],
      complexity: "High"
    },

    // Query-Specific Science Rubrics
    {
      id: "physics_problem_solving_rubric",
      name: "Physics Problem Solving Assistance",
      queryRelated: true,
      domain: "science",
      subdomain: "physics",
      language: "english",
      source: "educational",
      query: "A ball is thrown upward with an initial velocity of 20 m/s. How high will it go and how long will it take to return to the ground?",
      description: "Framework for evaluating physics problem-solving help and conceptual explanations.",
      scenario: "Physics tutoring, homework assistance, and concept clarification sessions",
      rubrics: [
        "Problem Analysis: Break down complex physics problems into manageable components."
      ],
      complexity: "High"
    },
    {
      id: "chemistry_experiment_guidance_rubric",
      name: "Chemistry Experiment Guidance",
      queryRelated: true,
      domain: "science",
      subdomain: "chemistry",
      language: "english",
      source: "academic",
      query: "What safety precautions should I take when performing a titration experiment with sulfuric acid?",
      description: "Evaluation criteria for providing chemistry experiment guidance and safety instruction.",
      scenario: "Laboratory assistance, experiment planning, and chemistry education support",
      rubrics: [
        "Safety First: Prioritize laboratory safety and proper handling procedures."
      ],
      complexity: "High"
    },

    // Query-Specific Engineering Rubrics
    {
      id: "system_design_consultation_rubric",
      name: "System Design Consultation",
      queryRelated: true,
      domain: "engineering",
      subdomain: "software",
      language: "english",
      source: "industry_standard",
      query: "How would you design a URL shortening service like bit.ly that can handle millions of requests per day?",
      description: "Evaluation criteria for providing system design advice and architectural guidance.",
      scenario: "System design interviews, architecture consulting, and technical decision support",
      rubrics: [
        "Requirements Analysis: Thoroughly understand and clarify system requirements and constraints.",
      ],
      complexity: "Very High"
    },

    // Query-Specific Math Rubrics
    {
      id: "calculus_tutoring_rubric",
      name: "Calculus Tutoring Effectiveness",
      queryRelated: true,
      domain: "math",
      subdomain: "calculus",
      language: "english",
      source: "educational",
      query: "I'm struggling to understand the concept of limits. Can you explain what lim(x→0) sin(x)/x equals and why?",
      description: "Specialized rubric for evaluating calculus tutoring interactions and problem-solving guidance.",
      scenario: "One-on-one tutoring sessions, homework help, and calculus concept explanation",
      rubrics: [
        "Adaptive Explanation: Adjust explanation complexity based on student's demonstrated understanding level.",
        "Conceptual Foundation: Build understanding from fundamental rubrics rather than just procedural steps."
      ],
      complexity: "High"
    }
  ];

  // —— Utils
  function show(el){ el.hidden = false; }
  function hide(el){ el.hidden = true; }
  function setLoading(on){
    on ? (show(elLoading), [elError, elCategories, elRubrics, elEmpty, elStats, elCrumb].forEach(hide))
       : hide(elLoading);
  }
  function setError(on){ on ? (show(elError), [elLoading].forEach(hide)) : hide(elError); }
  function clampTxt(s, n){ if(!s) return ""; return s.length<=n? s : s.slice(0,n)+"…"; }
  function debounce(fn, ms=250){ let t; return (...a)=>{ clearTimeout(t); t=setTimeout(()=>fn(...a), ms); }; }

  // —— Data Loading
  async function loadAll(){
    setLoading(true); setError(false);
    try{
      ALL_RUBRICS = MOCK_RUBRICS;
      if(!ALL_RUBRICS.length) throw new Error("no data");

      // Group rubrics by category -> domain -> subdomain
      GROUPED_RUBRICS = ALL_RUBRICS.reduce((acc, rubric)=>{
        const categoryKey = rubric.queryRelated ? "Query-Specific Rubrics" : "Query-Agnostic Rubrics";
        const domainKey = rubric.domain;
        const subdomainKey = rubric.subdomain || "general";

        if (!acc[categoryKey]) acc[categoryKey] = {};
        if (!acc[categoryKey][domainKey]) acc[categoryKey][domainKey] = {};
        if (!acc[categoryKey][domainKey][subdomainKey]) acc[categoryKey][domainKey][subdomainKey] = [];

        acc[categoryKey][domainKey][subdomainKey].push(rubric);
        return acc;
      }, {});
      renderCategories();
    }catch(e){
      setError(true);
    }finally{
      setLoading(false);
    }
  }

  // —— Render Categories (Top Level)
  function renderCategories(){
    VIEW = "categories";
    CURR_CATEGORY = null; CURR_DOMAIN = null; CURR_SUBDOMAIN = null;
    hide(elRubrics); hide(elEmpty); show(elCategories);
    hide(elCrumb);
    elCrumbTitle.textContent = "Rubric Library";
    elType.textContent = "domains";

    const sections = Object.entries(GROUPED_RUBRICS).map(([categoryName, domains])=>{
      const domainCards = Object.entries(domains).map(([domainName, subdomains])=>{
        const totalRubrics = Object.values(subdomains).reduce((sum, rubrics) => {
          return sum + (Array.isArray(rubrics) ? rubrics.length : 0);
        }, 0);

        const subdomainCount = Object.keys(subdomains).length;
        const hasSubdomains = subdomainCount > 1 || !subdomains.general;

        return `
          <div class="ml-card-item" data-category="${categoryName}" data-domain="${domainName}">
            <div class="ml-card-head">
              <div class="ml-card-left">
                <div class="ml-chip ${domainName}">${domainName.toUpperCase()}</div>
              </div>
            </div>
            <div class="ml-card-title-main">${domainName.charAt(0).toUpperCase() + domainName.slice(1)} Domain</div>
            <div class="ml-card-sub">${subdomainCount} ${subdomainCount > 1 ? 'subdomains' : 'subdomain'}</div>
            <div class="ml-card-sample">Specialized evaluation rubrics for ${domainName} domain tasks and content</div>
            <div class="ml-card-foot">
              <span style="opacity: 0.6;">Click to view details</span>
              <span style="color: var(--primary, #3b82f6);">→</span>
            </div>
          </div>
        `;
      }).join("");

      return `
        <section class="ml-section">
          <h3>${categoryName}</h3>
          <div class="ml-grid">
            ${domainCards}
          </div>
        </section>
      `;
    }).join("");

    elCategories.innerHTML = sections;
    bindDomainClicks();

    show(elStats);
    const totalDomains = Object.values(GROUPED_RUBRICS).reduce((sum, domains) => sum + Object.keys(domains).length, 0);
    elCount.textContent = totalDomains;
    elTotal.textContent = totalDomains;
  }

  // —— Render Domains (Second Level)
  function renderDomains(categoryName){
    VIEW = "domains";
    CURR_CATEGORY = categoryName; CURR_DOMAIN = null; CURR_SUBDOMAIN = null;
    hide(elRubrics); hide(elEmpty); show(elCategories);
    show(elCrumb);
    elCrumbTitle.textContent = categoryName;
    elType.textContent = "domains";

    const domains = GROUPED_RUBRICS[categoryName] || {};

    const sections = Object.entries(domains).map(([domainName, subdomains])=>{
      const totalRubrics = Object.values(subdomains).reduce((sum, rubrics) => {
        return sum + (Array.isArray(rubrics) ? rubrics.length : 0);
      }, 0);

      const subdomainCount = Object.keys(subdomains).length;
      const hasSubdomains = subdomainCount > 1 || !subdomains.general;

      return `
        <div class="ml-card-item" data-domain="${domainName}">
          <div class="ml-card-head">
            <div class="ml-card-left">
              <div class="ml-chip ${domainName}">${domainName.toUpperCase()}</div>
            </div>
          </div>
          <div class="ml-card-title-main">${domainName.charAt(0).toUpperCase() + domainName.slice(1)} Domain</div>
          <div class="ml-card-sub">${subdomainCount} ${subdomainCount > 1 ? 'subdomains' : 'subdomain'}</div>
          <div class="ml-card-sample">Specialized evaluation rubrics for ${domainName} domain tasks and content</div>
          <div class="ml-card-foot">
            <span style="opacity: 0.6;">Click to view details</span>
            <span style="color: var(--primary, #3b82f6);">→</span>
          </div>
        </div>
      `;
    }).join("");

    elCategories.innerHTML = `
      <section class="ml-section">
        <h3>${categoryName} - Domains</h3>
        <div class="ml-grid">
          ${sections}
        </div>
      </section>
    `;
    bindDomainClicks();

    show(elStats);
    elCount.textContent = Object.keys(domains).length;
    elTotal.textContent = Object.keys(domains).length;
  }

  // —— Render Subdomains (Third Level)
  function renderSubdomains(categoryName, domainName){
    VIEW = "subdomains";
    CURR_CATEGORY = categoryName; CURR_DOMAIN = domainName; CURR_SUBDOMAIN = null;
    show(elCrumb);
    elCrumbTitle.textContent = `${categoryName} > ${domainName}`;

    const subdomains = GROUPED_RUBRICS[categoryName][domainName] || {};

    // If only one subdomain (general), go directly to rubrics
    if (Object.keys(subdomains).length === 1 && subdomains.general) {
      renderRubrics(categoryName, domainName, "general");
      return;
    }

    // For Query-Agnostic: show subdomains as cards that lead to rubric lists
    if (categoryName === "Query-Agnostic Rubrics") {
      hide(elRubrics); hide(elEmpty); show(elCategories);
      elType.textContent = "subdomains";

      const sections = Object.entries(subdomains).map(([subdomainName, rubrics])=>{
        const rubricCount = Array.isArray(rubrics) ? rubrics.length : 0;

        return `
          <div class="ml-card-item" data-category="${categoryName}" data-domain="${domainName}" data-subdomain="${subdomainName}">
            <div class="ml-card-head">
              <div class="ml-card-left">
                <div class="ml-chip ${subdomainName}">${subdomainName.toUpperCase()}</div>
              </div>
            </div>
            <div class="ml-card-title-main">${subdomainName.charAt(0).toUpperCase() + subdomainName.slice(1)}</div>
            <div class="ml-card-sub">${rubricCount} evaluation rubrics</div>
            <div class="ml-card-sample">Evaluation rubrics specialized for ${subdomainName} development and assessment</div>
            <div class="ml-card-foot">
              <span style="opacity: 0.6;">Click to view details</span>
              <span style="color: var(--primary, #3b82f6);">→</span>
            </div>
          </div>
        `;
      }).join("");

      elCategories.innerHTML = `
        <section class="ml-section">
          <h3>${domainName.charAt(0).toUpperCase() + domainName.slice(1)} Subdomains</h3>
          <div class="ml-grid">
            ${sections}
          </div>
        </section>
      `;
      bindSubdomainClicks();

      show(elStats);
      elCount.textContent = Object.keys(subdomains).length;
      elTotal.textContent = Object.keys(subdomains).length;
    }
    // For Query-Specific: show all rubrics in grid layout
    else {
      hide(elCategories); hide(elEmpty); show(elRubrics);
      elType.textContent = "rubrics";

      // Flatten all rubrics from all subdomains
      const allRubrics = Object.entries(subdomains).flatMap(([subdomainName, rubrics]) =>
        Array.isArray(rubrics) ? rubrics.map(r => ({...r, displaySubdomain: subdomainName})) : []
      );

      if(!allRubrics.length){
        hide(elRubrics); show(elEmpty); hide(elStats); return;
      }

      // Grid layout like general domain
      elRubrics.innerHTML = allRubrics.map((rubric, idx)=>`
        <div class="ml-card-item" data-idx="${idx}">
          <div class="ml-card-head">
            <div class="ml-card-left">
              <div class="ml-chip query-specific">QUERY-SPECIFIC</div>
              <div class="ml-chip ${getComplexityClass(rubric.complexity)}">${rubric.complexity.toUpperCase()}</div>
            </div>
          </div>
          <div class="ml-card-title-main">${rubric.name}</div>
          <div class="ml-card-sub">${rubric.domain}${rubric.displaySubdomain ? ` > ${rubric.displaySubdomain}` : ''}</div>
          <div class="ml-card-sample">${clampTxt(rubric.description, 120)}</div>
          <div class="ml-card-foot">
            <span style="opacity: 0.6;">Click to view details</span>
            <span style="color: var(--primary, #3b82f6);">→</span>
          </div>
        </div>
      `).join("");

      // Modal binding
      [...elRubrics.querySelectorAll(".ml-card-item")].forEach(card=>{
        card.addEventListener("click", ()=>{
          const idx = Number(card.getAttribute("data-idx"));
          const rubric = allRubrics[idx];
          showRubricModal(rubric);
        });
      });

      show(elStats);
      elCount.textContent = allRubrics.length;
      elTotal.textContent = allRubrics.length;
    }
  }

  // —— Render Rubrics (Final Level)
  function renderRubrics(categoryName, domainName, subdomainName){
    VIEW = "rubrics";
    CURR_CATEGORY = categoryName; CURR_DOMAIN = domainName; CURR_SUBDOMAIN = subdomainName;
    hide(elCategories); hide(elEmpty); show(elRubrics);
    show(elCrumb);
    elType.textContent = "rubrics";
    // Avoid showing duplicate names in breadcrumb (e.g., general > general)
    const breadcrumb = domainName === subdomainName
      ? `${categoryName} > ${domainName}`
      : `${categoryName} > ${domainName} > ${subdomainName}`;
    elCrumbTitle.textContent = breadcrumb;

    const rubricList = GROUPED_RUBRICS[categoryName]?.[domainName]?.[subdomainName] || [];

    if(!rubricList.length){
      hide(elRubrics); show(elEmpty); hide(elStats); return;
    }

    elRubrics.innerHTML = rubricList.map((rubric, idx)=>`
      <div class="ml-card-item" data-idx="${idx}">
        <div class="ml-card-head">
          <div class="ml-card-left">
            <div class="ml-chip ${rubric.queryRelated ? 'query-specific' : 'query-agnostic'}">${rubric.queryRelated ? 'QUERY-SPECIFIC' : 'QUERY-AGNOSTIC'}</div>
            <div class="ml-chip ${getComplexityClass(rubric.complexity)}">${rubric.complexity.toUpperCase()}</div>
          </div>
        </div>
        <div class="ml-card-title-main">${rubric.name}</div>
        <div class="ml-card-sub">${rubric.domain}${rubric.subdomain ? ` > ${rubric.subdomain}` : ''}</div>
        <div class="ml-card-sample">${clampTxt(rubric.description, 120)}</div>
        <div class="ml-card-foot">
          <span style="opacity: 0.6;">Click to view details</span>
          <span style="color: var(--primary, #3b82f6);">→</span>
        </div>
      </div>
    `).join("");

    // Modal binding
    [...elRubrics.querySelectorAll(".ml-card-item")].forEach(card=>{
      card.addEventListener("click", ()=>{
        const idx = Number(card.getAttribute("data-idx"));
        const rubric = rubricList[idx];
        showRubricModal(rubric);
      });
    });

    show(elStats);
    elCount.textContent = rubricList.length;
    elTotal.textContent = rubricList.length;
  }

  function getComplexityClass(complexity) {
    switch(complexity) {
      case 'Low': return 'success';
      case 'Medium': return 'warning';
      case 'High': case 'Very High': return 'danger';
      default: return 'success';
    }
  }

  function showRubricModal(rubric) {
    mCategory.textContent = rubric.queryRelated ? "Query-Specific" : "Query-Agnostic";
    mCategory.className = `ml-chip ${rubric.queryRelated ? 'query-specific' : 'query-agnostic'}`;
    mDomain.textContent = `${rubric.domain}${rubric.subdomain ? ` > ${rubric.subdomain}` : ''}`;

    // Show query for Query-Specific rubrics
    if (rubric.queryRelated && rubric.query) {
      mQuery.textContent = rubric.query;
      mQuerySection.hidden = false;
    } else {
      mQuerySection.hidden = true;
    }

    // Hide description and scenario for Query-Specific rubrics
    if (rubric.queryRelated) {
      mDescriptionSection.hidden = true;
      mScenarioSection.hidden = true;
    } else {
      mDescriptionSection.hidden = false;
      mScenarioSection.hidden = false;
      mDescription.textContent = rubric.description;
      mScenario.textContent = rubric.scenario;
    }

    // Handle rubrics
    if (rubric.rubrics && rubric.rubrics.length > 0) {
      const rubricsList = rubric.rubrics.map((rubric, idx) =>
        `<div class="rubric-item">
          <span class="rubric-number">P${idx + 1}</span>
          <div class="rubric-content">${rubric}</div>
        </div>`
      ).join("");
      mrubrics.innerHTML = `<div class="rubric-list">${rubricsList}</div>`;
    } else {
      mrubrics.innerHTML = '<div class="ml-muted">No specific rubrics defined</div>';
    }

    // Usage example
    const usageExample = `from rm_gallery.core.reward import BaseListWiserubricReward
from rm_gallery.core.model.openai_llm import OpenaiLLM

# Create reward model with this rubric
llm = OpenaiLLM(model="qwen3-8b", enable_thinking=True)
reward = BaseListWiserubricReward(
    name="${rubric.id}",
    desc="${rubric.description}",
    scenario="${rubric.scenario}",
    rubrics=${JSON.stringify(rubric.rubrics || [])},
    llm=llm
)

# Use the reward model
result = reward.evaluate(sample)`;
    mUsage.textContent = usageExample;

    // Rubric info
    mId.textContent = rubric.id;
    mDomainInfo.textContent = `${rubric.domain}${rubric.subdomain ? ` > ${rubric.subdomain}` : ''}`;
    mLanguage.textContent = rubric.language;
    mSource.textContent = rubric.source;
    mrubricCount.textContent = rubric.rubrics ? rubric.rubrics.length : 0;
    mComplexity.textContent = rubric.complexity;

    dlg.showModal();
  }

  function bindCategoryClicks(){
    [...elCategories.querySelectorAll(".ml-card-item[data-category]")].forEach(card=>{
      card.addEventListener("click", ()=>{
        const categoryName = card.getAttribute("data-category");
        renderDomains(categoryName);
      });
    });
  }

  function bindDomainClicks(){
    [...elCategories.querySelectorAll(".ml-card-item[data-domain]")].forEach(card=>{
      card.addEventListener("click", ()=>{
        const categoryName = card.getAttribute("data-category");
        const domainName = card.getAttribute("data-domain");

        // For Query-Agnostic: check if domain has multiple subdomains
        if (categoryName === "Query-Agnostic Rubrics") {
          const subdomains = GROUPED_RUBRICS[categoryName][domainName] || {};
          const subdomainKeys = Object.keys(subdomains);

          // If only general subdomain or single subdomain, go directly to rubrics
          if (subdomainKeys.length === 1) {
            renderRubrics(categoryName, domainName, subdomainKeys[0]);
          } else {
            // Multiple subdomains, show subdomain selection
            renderSubdomains(categoryName, domainName);
          }
        } else {
          // For Query-Specific: always show Hugging Face style
          renderSubdomains(categoryName, domainName);
        }
      });
    });
  }

  function bindSubdomainClicks(){
    [...elCategories.querySelectorAll(".ml-card-item[data-subdomain]")].forEach(card=>{
      card.addEventListener("click", ()=>{
        const categoryName = card.getAttribute("data-category");
        const domainName = card.getAttribute("data-domain");
        const subdomainName = card.getAttribute("data-subdomain");
        renderRubrics(categoryName, domainName, subdomainName);
      });
    });
  }

  // —— Search
  function handleSearch(){
    const q = elSearch.value.trim().toLowerCase();
    if(!q){
      // Clear search: return to categories view
      renderCategories();
      return;
    }

    // Global search across all rubrics
    const filteredRubrics = ALL_RUBRICS.filter(rubric =>
      rubric.name.toLowerCase().includes(q) ||
      rubric.description.toLowerCase().includes(q) ||
      rubric.domain.toLowerCase().includes(q) ||
      (rubric.subdomain && rubric.subdomain.toLowerCase().includes(q)) ||
      rubric.language.toLowerCase().includes(q) ||
      rubric.source.toLowerCase().includes(q) ||
      (rubric.rubrics && rubric.rubrics.some(p => p.toLowerCase().includes(q)))
    );

    // Show search results as rubrics (keep original VIEW state for restoration)
    const PREV_VIEW = VIEW;
    const PREV_CATEGORY = CURR_CATEGORY;
    const PREV_DOMAIN = CURR_DOMAIN;
    const PREV_SUBDOMAIN = CURR_SUBDOMAIN;

    VIEW = "search";
    CURR_CATEGORY = null; CURR_DOMAIN = null; CURR_SUBDOMAIN = null;
    hide(elCategories); hide(elEmpty); show(elRubrics);
    show(elCrumb);
    elType.textContent = "search results";
    elCrumbTitle.textContent = `Search: "${q}"`;

    if(!filteredRubrics.length){
      hide(elRubrics); show(elEmpty); hide(elStats); return;
    }

    elRubrics.innerHTML = filteredRubrics.map((rubric, idx)=>`
      <div class="ml-card-item" data-idx="${idx}">
        <div class="ml-card-head">
          <div class="ml-card-left">
            <div class="ml-chip ${rubric.queryRelated ? 'query-specific' : 'query-agnostic'}">${rubric.queryRelated ? 'QUERY-SPECIFIC' : 'QUERY-AGNOSTIC'}</div>
            <div class="ml-chip ${getComplexityClass(rubric.complexity)}">${rubric.complexity.toUpperCase()}</div>
          </div>
        </div>
        <div class="ml-card-title-main">${rubric.name}</div>
        <div class="ml-card-sub">${rubric.domain}${rubric.subdomain ? ` > ${rubric.subdomain}` : ''}</div>
        <div class="ml-card-sample">${clampTxt(rubric.description, 120)}</div>
        <div class="ml-card-foot">
          <span style="opacity: 0.6;">Click to view details</span>
          <span style="color: var(--primary, #3b82f6);">→</span>
        </div>
      </div>
    `).join("");

    // Modal binding for search results
    [...elRubrics.querySelectorAll(".ml-card-item")].forEach(card=>{
      card.addEventListener("click", ()=>{
        const idx = Number(card.getAttribute("data-idx"));
        const rubric = filteredRubrics[idx];
        showRubricModal(rubric);
      });
    });

    show(elStats);
    elCount.textContent = filteredRubrics.length;
    elTotal.textContent = filteredRubrics.length;
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