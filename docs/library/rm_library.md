---
new: true
---
# RM Library

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
      <button class="ml-close" aria-label="Close">✕</button>
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
  --ml-gap: 1rem;
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
@media (min-width: 640px){ .ml-grid{ grid-template-columns: repeat(2, minmax(0,1fr)); } }
@media (min-width: 1024px){ .ml-grid{ grid-template-columns: repeat(3, minmax(0,1fr)); } }

/* categories stacked */
.ml-stacked { display: grid; gap: 1.25rem; }
.ml-section{ display:grid; gap:.5rem; }
.ml-section h3{ margin:.25rem 0; font-size:1.05rem; font-weight:700; opacity:.85; display:flex; gap:.5rem; align-items:center; }

.ml-card-item{
  background: var(--card, var(--background, #fff));
  border: 1px solid var(--border, rgba(0,0,0,.08));
  border-radius: var(--ml-radius);
  padding: 1rem;
  transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
  cursor: pointer;
}
.ml-card-item:hover{
  transform: translateY(-2px);
  box-shadow: var(--ml-shadow);
  border-color: var(--primary, #3b82f6);
}
.ml-card-head{ display:flex; align-items:flex-start; justify-content:space-between; gap:.75rem; margin-bottom:.5rem; }
.ml-card-title{ font-weight: 650; font-size: 1rem; }
.ml-card-sub{ font-size: .85rem; opacity: .7; }
.ml-card-sample{ margin-top:.5rem; font-size:.92rem; line-height:1.5; opacity:.9; display:-webkit-box; -webkit-line-clamp:3; -webkit-box-orient:vertical; overflow:hidden; }
.ml-card-foot{ display:flex; justify-content:space-between; align-items:center; border-top:1px solid var(--border, rgba(0,0,0,.08)); padding-top:.5rem; margin-top:.75rem; font-size:.85rem; opacity:.8; }

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
  padding:.55rem .9rem; border-radius:.55rem; cursor:pointer;
}
.ml-btn.secondary{ background: var(--muted, rgba(0,0,0,.03)); }
.ml-btn:hover{ border-color: var(--primary, #3b82f6); }

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
.ml-chip{ display:inline-block; padding:.25rem .55rem; border-radius:999px; font-size:.78rem;
  background: color-mix(in srgb, var(--primary,#3b82f6) 12%, transparent); color: var(--primary,#3b82f6);
}
.ml-chip.success{
  background: color-mix(in srgb, #16a34a 14%, transparent);
  color: #16a34a;
}
.ml-chip.warning{
  background: color-mix(in srgb, #f59e0b 14%, transparent);
  color: #b45309;
}
.ml-chip.alignment {
  background: color-mix(in srgb, #8b5cf6 14%, transparent);
  color: #7c3aed;
}
.ml-chip.code {
  background: color-mix(in srgb, #06b6d4 14%, transparent);
  color: #0891b2;
}
.ml-chip.math {
  background: color-mix(in srgb, #f59e0b 14%, transparent);
  color: #d97706;
}
.ml-chip.format {
  background: color-mix(in srgb, #10b981 14%, transparent);
  color: #059669;
}
.ml-chip.general {
  background: color-mix(in srgb, #6b7280 14%, transparent);
  color: #4b5563;
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
}
.ml-meta span{ opacity:.7; }
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
.ml-modal-header{ display:flex; justify-content:space-between; align-items:center; gap:.75rem; margin-bottom:.5rem; }
.ml-close{ border:none; background:none; font-size:1.1rem; cursor:pointer; opacity:.6; }
.ml-close:hover{ opacity:1; }
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
    "Alignment": ["alignment"],
    "Code": ["code"],
    "Math": ["math"],
    "Format & Style": ["format"],
    "General": ["general"]
  };

  // —— Mock RM Data (In real implementation, this would come from the backend)
  const MOCK_RMS = [
    {
      name: "chat_listwise_reward",
      class_name: "ChatListWiseReward",
      category: "alignment",
      reward_type: "ListWise",
      description: "Chat: Simulates human conversation and communicates a variety of topics through text understanding and generation, emphasizing coherence and natural flow of interaction.",
      scenario: "Chat: Simulates human conversation and communicates a variety of topics through text understanding and generation, emphasizing coherence and natural flow of interaction.",
      principles: [
        "Address Core Argument/Intent Directly: Prioritize engaging with the user's central claim, perspective, or question explicitly, ensuring responses align with their stated goals or concerns rather than diverging into tangential topics.",
        "Provide Actionable, Context-Specific Guidance: Offer concrete, practical steps or solutions tailored to the user's unique situation, balancing clarity with adaptability to empower informed decisions or actions.",
        "Ensure Factual Accuracy and Contextual Nuance: Correct misconceptions, clarify complexities, and ground responses in precise details or evidence while avoiding oversimplification or speculative interpretations."
      ],
      module_path: "rm_gallery.gallery.rm.alignment.helpfulness.chat"
    },
    {
      name: "math_verify_reward",
      class_name: "MathVerifyReward",
      category: "math",
      reward_type: "PointWise",
      description: "Verifies mathematical expressions using the math_verify library, supporting both LaTeX and plain expressions",
      scenario: "Mathematical problem solving and verification tasks",
      principles: null,
      module_path: "rm_gallery.gallery.rm.math.math"
    },
    {
      name: "code_syntax_check",
      class_name: "SyntaxCheckReward",
      category: "code",
      reward_type: "PointWise",
      description: "Check code syntax using Abstract Syntax Tree to validate Python code blocks.",
      scenario: "Python code generation and validation tasks",
      principles: null,
      module_path: "rm_gallery.gallery.rm.code.code"
    },
    {
      name: "code_style",
      class_name: "CodeStyleReward",
      category: "code",
      reward_type: "PointWise",
      description: "Basic code style checking including indentation consistency and naming conventions.",
      scenario: "Python code style and formatting evaluation",
      principles: null,
      module_path: "rm_gallery.gallery.rm.code.code"
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
      const itemsHtml = categories.map(category=>{
        const rms = GROUPED_RMS[category];
        const sampleRM = rms[0] || {};
        const description = sampleRM.description || "No description available";

        return `
          <div class="ml-card-item" data-category="${category}">
            <div class="ml-card-head">
              <div>
                <div class="ml-card-title">${categoryName}</div>
                <div class="ml-card-sub">${rms.length} reward models</div>
              </div>
              <div class="ml-chip ${category}">${category.toUpperCase()}</div>
            </div>
            <div class="ml-card-sample">${clampTxt(description, 150)}</div>
            <div class="ml-card-foot">
              <span>📊 ${rms.length} models</span>
              <span>Explore →</span>
            </div>
          </div>
        `;
      }).join("");

      return categories.length ? `
      <section class="ml-section">
        <h3>${categoryName}</h3>
        <div class="ml-grid">
          ${itemsHtml}
        </div>
      </section>
      ` : "";
    }).join("");

    elCategories.innerHTML = sections;
    bindCategoryClicks();

    show(elStats);
    const totalRMs = ALL_RMS.length;
    elCount.textContent = totalRMs;
    elTotal.textContent = totalRMs;
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
          <div class="ml-chip ${rm.reward_type === 'ListWise' ? 'success' : 'warning'}">${rm.reward_type}</div>
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

  function bindCategoryClicks(){
    [...elCategories.querySelectorAll(".ml-card-item[data-category]")].forEach(card=>{
      card.addEventListener("click", ()=>{
        CURR_CATEGORY = card.getAttribute("data-category");
        renderModels(GROUPED_RMS[CURR_CATEGORY]);
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
  elRetry?.addEventListener("click", loadAll);
  elBack?.addEventListener("click", ()=> renderCategories());
  elSearch?.addEventListener("input", debounce(handleSearch, 250));
  elClear?.addEventListener("click", ()=>{
    elSearch.value = ""; handleSearch();
  });

  // —— Init
  document.addEventListener("DOMContentLoaded", loadAll);
})();
</script>