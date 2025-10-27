/**
 * Navigation Scroll Position Fix
 * Preserves sidebar scroll position when clicking navigation items
 */

(function() {
    'use strict';

    const STORAGE_KEY = 'nav-scroll-position';
    const NAV_SELECTORS = [
        '.md-sidebar--primary .md-sidebar__scrollwrap',
        '.md-nav--primary',
        'nav[data-md-component="sidebar"]',
        '.md-sidebar.md-sidebar--primary',
        '[data-sidebar="content"]',
        '[data-slot="sidebar-content"]',
        '[data-slot="sidebar-wrapper"] [data-slot="sidebar"] [data-slot="sidebar-content"]'
    ];

    function getNavElement() {
        for (const selector of NAV_SELECTORS) {
            const elem = document.querySelector(selector);
            if (elem) {
                return elem;
            }
        }
        return null;
    }

    function saveScrollPosition() {
        const navElement = getNavElement();
        if (navElement) {
            const scrollPosition = navElement.scrollTop;
            sessionStorage.setItem(STORAGE_KEY, scrollPosition.toString());
        }
    }

    function restoreScrollPosition(immediate = false) {
        const navElement = getNavElement();
        const savedPosition = sessionStorage.getItem(STORAGE_KEY);

        if (navElement && savedPosition) {
            const scrollPos = parseInt(savedPosition, 10);

            if (immediate) {
                // Restore immediately without delay to prevent flashing
                navElement.scrollTop = scrollPos;
            } else {
                // Use requestAnimationFrame for smooth restoration
                requestAnimationFrame(() => {
                    navElement.scrollTop = scrollPos;
                });
            }
        }
    }

    let lastActiveLink = null;
    let lastNavigationTime = 0;

    function onNavClick(event) {
        const target = event.currentTarget;
        lastActiveLink = target;
        const now = Date.now();
        lastNavigationTime = now;
        saveScrollPosition();
    }

    function getNavLinks() {
        return document.querySelectorAll('.md-nav__link, [data-sidebar="menu-button"], [data-slot="sidebar-menu-button"] a, [data-slot="sidebar-menu-button"], [data-sidebar="menu-button"]');
    }

    function attachScrollSaver() {
        const navLinks = getNavLinks();

        navLinks.forEach(link => {
            link.removeEventListener('click', onNavClick);
            link.addEventListener('click', onNavClick);
        });
    }

    // Initialize on page load
    function init(immediate = false) {
        // Restore scroll position
        restoreScrollPosition(immediate);

        // Attach scroll savers to navigation links
        attachScrollSaver();

        // Also save on page unload
        window.removeEventListener('beforeunload', saveScrollPosition);
        window.addEventListener('beforeunload', saveScrollPosition);

        // Handle dynamic content loading (for MkDocs instant loading)
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                if (mutation.addedNodes.length) {
                    attachScrollSaver();
                }
            });
        });

        // Observe the navigation for changes
        const nav = document.querySelector('.md-sidebar--primary');
        const navWrapper = document.querySelector('[data-slot="sidebar-wrapper"]');
        if (navWrapper) {
            observer.observe(navWrapper, {
                childList: true,
                subtree: true
            });
        }
    }

    // Restore scroll position immediately on script load to prevent flashing
    // This runs before the page is fully rendered
    (function earlyRestore() {
        const navElement = getNavElement();
        const savedPosition = sessionStorage.getItem(STORAGE_KEY);
        if (navElement && savedPosition) {
            navElement.scrollTop = parseInt(savedPosition, 10);
        }
    })();

    // Run when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => init(true));
    } else {
        init(true);
    }

    // Handle instant loading in MkDocs Material theme
    document.addEventListener('DOMContentSwitch', () => {
        // Restore immediately to prevent flash
        restoreScrollPosition(true);
        // Then initialize event handlers
        setTimeout(() => {
            attachScrollSaver();
        }, 50);
    });

    document.addEventListener('navigation', () => {
        // Restore immediately on navigation
        restoreScrollPosition(true);
        // Then reinitialize
        setTimeout(() => {
            attachScrollSaver();
        }, 50);
    });

})();

