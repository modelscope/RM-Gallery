/**
 * Code Block Zoom Control
 * Adds zoom in/out controls to code blocks for better readability
 */

(function() {
    'use strict';

    // Font size presets
    const FONT_SIZES = {
        'small': '0.85em',
        'medium': '1em',
        'large': '1.15em',
        'xlarge': '1.3em'
    };

    // Default font size
    let currentSize = localStorage.getItem('codeBlockFontSize') || 'medium';

    function createZoomControl() {
        const control = document.createElement('div');
        control.className = 'code-zoom-control';
        control.innerHTML = `
            <div class="zoom-control-container">
                <button class="zoom-btn zoom-decrease" title="Decrease code font size" aria-label="Decrease font size">A-</button>
                <button class="zoom-btn zoom-reset" title="Reset code font size" aria-label="Reset font size">A</button>
                <button class="zoom-btn zoom-increase" title="Increase code font size" aria-label="Increase font size">A+</button>
            </div>
        `;

        return control;
    }

    function updateCodeBlockSize(size) {
        const fontSize = FONT_SIZES[size];
        const codeBlocks = document.querySelectorAll('.highlight code, .jp-InputArea-editor code');

        codeBlocks.forEach(block => {
            block.style.fontSize = fontSize;
        });

        currentSize = size;
        localStorage.setItem('codeBlockFontSize', size);
    }

    function initZoomControls() {
        // Check if controls already exist
        if (document.querySelector('.code-zoom-control')) {
            return;
        }

        // Create and add zoom control to the page
        const control = createZoomControl();

        // Try to add to navigation or create a fixed position control
        const nav = document.querySelector('.md-header__inner') || document.body;

        if (nav === document.body) {
            control.style.position = 'fixed';
            control.style.bottom = '20px';
            control.style.right = '20px';
            control.style.zIndex = '1000';
        }

        nav.appendChild(control);

        // Add event listeners
        const decreaseBtn = control.querySelector('.zoom-decrease');
        const resetBtn = control.querySelector('.zoom-reset');
        const increaseBtn = control.querySelector('.zoom-increase');

        decreaseBtn.addEventListener('click', () => {
            const sizes = Object.keys(FONT_SIZES);
            const currentIndex = sizes.indexOf(currentSize);
            if (currentIndex > 0) {
                updateCodeBlockSize(sizes[currentIndex - 1]);
            }
        });

        resetBtn.addEventListener('click', () => {
            updateCodeBlockSize('medium');
        });

        increaseBtn.addEventListener('click', () => {
            const sizes = Object.keys(FONT_SIZES);
            const currentIndex = sizes.indexOf(currentSize);
            if (currentIndex < sizes.length - 1) {
                updateCodeBlockSize(sizes[currentIndex + 1]);
            }
        });

        // Apply saved size
        updateCodeBlockSize(currentSize);
    }

    function addZoomStyles() {
        const styleId = 'code-zoom-styles';

        // Don't add styles if already present
        if (document.getElementById(styleId)) {
            return;
        }

        const style = document.createElement('style');
        style.id = styleId;
        style.textContent = `
            .code-zoom-control {
                display: inline-flex;
                align-items: center;
                gap: 4px;
                padding: 4px;
                background: rgba(255, 255, 255, 0.95);
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
                transition: all 0.3s ease;
            }

            [data-md-color-scheme="slate"] .code-zoom-control {
                background: rgba(22, 27, 34, 0.95);
            }

            .zoom-control-container {
                display: flex;
                gap: 2px;
            }

            .zoom-btn {
                padding: 6px 10px;
                background: #f6f8fa;
                border: 1px solid #e1e4e8;
                border-radius: 6px;
                cursor: pointer;
                font-family: 'JetBrains Mono', monospace;
                font-size: 14px;
                font-weight: 600;
                color: #24292e;
                transition: all 0.2s ease;
                min-width: 36px;
                display: flex;
                align-items: center;
                justify-content: center;
            }

            [data-md-color-scheme="slate"] .zoom-btn {
                background: #161b22;
                border-color: #30363d;
                color: #c9d1d9;
            }

            .zoom-btn:hover {
                background: #e1e4e8;
                transform: translateY(-1px);
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }

            [data-md-color-scheme="slate"] .zoom-btn:hover {
                background: #1c2128;
            }

            .zoom-btn:active {
                transform: translateY(0);
            }

            .zoom-reset {
                border-left: 1px solid #d1d5da;
                border-right: 1px solid #d1d5da;
            }

            [data-md-color-scheme="slate"] .zoom-reset {
                border-left-color: #30363d;
                border-right-color: #30363d;
            }

            @media (max-width: 768px) {
                .code-zoom-control {
                    position: fixed !important;
                    bottom: 60px !important;
                    right: 10px !important;
                    z-index: 1000;
                }
            }
        `;

        document.head.appendChild(style);
    }

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
            addZoomStyles();
            initZoomControls();
        });
    } else {
        addZoomStyles();
        initZoomControls();
    }

    // Re-initialize on navigation (for SPA-style navigation in MkDocs)
    document.addEventListener('navigation', () => {
        setTimeout(() => {
            updateCodeBlockSize(currentSize);
        }, 100);
    });

    // Keyboard shortcuts for zoom
    document.addEventListener('keydown', (event) => {
        // Ctrl/Cmd + Plus: Increase font size
        if ((event.ctrlKey || event.metaKey) && event.key === '+') {
            const sizes = Object.keys(FONT_SIZES);
            const currentIndex = sizes.indexOf(currentSize);
            if (currentIndex < sizes.length - 1) {
                updateCodeBlockSize(sizes[currentIndex + 1]);
                event.preventDefault();
            }
        }

        // Ctrl/Cmd + Minus: Decrease font size
        if ((event.ctrlKey || event.metaKey) && event.key === '-') {
            const sizes = Object.keys(FONT_SIZES);
            const currentIndex = sizes.indexOf(currentSize);
            if (currentIndex > 0) {
                updateCodeBlockSize(sizes[currentIndex - 1]);
                event.preventDefault();
            }
        }

        // Ctrl/Cmd + 0: Reset font size
        if ((event.ctrlKey || event.metaKey) && event.key === '0') {
            updateCodeBlockSize('medium');
            event.preventDefault();
        }
    });

})();

