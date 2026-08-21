"""Browser-side repair for Streamlit's persisted sidebar state.

EvidenceFlow uses Streamlit's native sidebar. The browser bridge has two jobs:

1. On the first visit in a browser tab, clear stale sidebar layout preferences
   that can leave a hosted/fullscreen Streamlit app collapsed. The page is
   reloaded once so ``initial_sidebar_state="expanded"`` can take effect.
2. Guarantee a working reopen affordance once the sidebar is collapsed.

On (2): earlier versions of this fix tried to force Streamlit's own native
reopen control visible via CSS/JS style overrides. That approach depends on
guessing which exact data-testid and DOM ancestry the installed Streamlit
build uses for that control, and silently breaks if the guess is wrong or an
ancestor element (not just the control itself) is what's hidden/clipped —
which is very plausibly what's happening here, since two rounds of
selector/!important fixes on the control itself did not resolve it.

Instead, this renders a small button that WE fully own (appended directly to
the parent document body, entirely outside Streamlit's own component tree,
so Streamlit's re-renders never touch or remove it) and is always visible
when the sidebar is collapsed, regardless of anything wrong with Streamlit's
native control. Clicking it fires a synthetic click on whichever native
toggle element currently exists in the DOM. A synthetic .click() on a
<button> fires its handlers even if that button is currently invisible,
zero-size, or nested inside a hidden ancestor — as long as it is still
present in the DOM and not disabled — so this works even if the native
control's visibility is broken for a reason we haven't identified.
"""
from __future__ import annotations

import streamlit.components.v1 as components


_TOGGLE_HTML = r"""
<script>
(() => {
  const parentWindow = window.parent;
  const doc = parentWindow.document;

  // Streamlit persists sidebar/layout preferences in browser storage.  A stale
  // collapsed preference can survive a deployment and can reproduce the
  // hosted-only behaviour where a fresh local run starts expanded but the same
  // public URL starts collapsed.  Reset sidebar-related preferences once per
  // browser-tab session, then reload so Streamlit applies initial_sidebar_state.
  const RECOVERY_FLAG = "evidenceflow-sidebar-recovery-v1";

  const recoverPersistedSidebarState = () => {
    try {
      if (parentWindow.sessionStorage.getItem(RECOVERY_FLAG) === "1") {
        return;
      }

      const staleKeys = [];
      for (let i = 0; i < parentWindow.localStorage.length; i += 1) {
        const key = parentWindow.localStorage.key(i);
        if (key && /sidebar/i.test(key)) {
          staleKeys.push(key);
        }
      }

      parentWindow.sessionStorage.setItem(RECOVERY_FLAG, "1");

      if (staleKeys.length > 0) {
        staleKeys.forEach((key) => parentWindow.localStorage.removeItem(key));
        parentWindow.location.reload();
        return true;
      }
    } catch (_) {
      // Storage access can be blocked by privacy settings. In that case the
      // native control repair below still runs and the app remains usable.
    }
    return false;
  };

  if (recoverPersistedSidebarState()) {
    return;
  }

  const NATIVE_SELECTORS = [
    '[data-testid="stSidebarCollapsedControl"] button',
    '[data-testid="stSidebarCollapsedControl"]',
    '[data-testid="stSidebarCollapseButton"] button',
    '[data-testid="stSidebarCollapseButton"]',
    '[data-testid="collapsedControl"] button',
    '[data-testid="collapsedControl"]'
  ];

  const findNativeToggle = () => {
    for (const selector of NATIVE_SELECTORS) {
      const el = doc.querySelector(selector);
      if (el) return el;
    }
    return null;
  };

  const isSidebarCollapsed = () => {
    const sidebar = doc.querySelector('section[data-testid="stSidebar"]');
    // aria-expanded is the accessibility attribute Streamlit sets on the
    // sidebar itself to reflect collapsed/expanded state; it is far less
    // likely to change across Streamlit versions than internal testids.
    if (sidebar && sidebar.getAttribute('aria-expanded') === 'false') return true;
    if (sidebar && sidebar.getAttribute('aria-expanded') === 'true') return false;
    // Fallback: no sidebar element found in the DOM at all reads as collapsed.
    return !sidebar;
  };

  const BUTTON_ID = 'evidenceflow-sidebar-reopen';

  const ensureCustomButton = () => {
    let btn = doc.getElementById(BUTTON_ID);
    const collapsed = isSidebarCollapsed();

    if (!collapsed) {
      if (btn) btn.style.setProperty('display', 'none', 'important');
      return;
    }

    if (!btn) {
      btn = doc.createElement('button');
      btn.id = BUTTON_ID;
      btn.type = 'button';
      btn.setAttribute('aria-label', 'Open sidebar');
      btn.innerHTML = '&#8250;';
      btn.addEventListener('click', () => {
        const native = findNativeToggle();
        if (native) {
          native.click();
        } else {
          // Last-resort fallback if no native control exists in the DOM at
          // all: reload once, which re-applies initial_sidebar_state.
          parentWindow.location.reload();
        }
      });
      doc.body.appendChild(btn);
    }

    // Inline styles here always take priority: this element is not managed
    // by Streamlit's own stylesheet, so there is nothing else in the cascade
    // to lose to, and !important keeps it that way defensively regardless.
    btn.style.setProperty('position', 'fixed', 'important');
    btn.style.setProperty('top', '0.6rem', 'important');
    btn.style.setProperty('left', '0.6rem', 'important');
    btn.style.setProperty('z-index', '999999', 'important');
    btn.style.setProperty('display', 'flex', 'important');
    btn.style.setProperty('align-items', 'center', 'important');
    btn.style.setProperty('justify-content', 'center', 'important');
    btn.style.setProperty('width', '2.2rem', 'important');
    btn.style.setProperty('height', '2.2rem', 'important');
    btn.style.setProperty('border-radius', '0.6rem', 'important');
    btn.style.setProperty('border', '1px solid rgba(120,120,120,0.35)', 'important');
    btn.style.setProperty('background', 'rgba(38,39,48,0.85)', 'important');
    btn.style.setProperty('color', '#fafafa', 'important');
    btn.style.setProperty('font-size', '1.2rem', 'important');
    btn.style.setProperty('line-height', '1', 'important');
    btn.style.setProperty('cursor', 'pointer', 'important');
    btn.style.setProperty('box-shadow', '0 4px 14px rgba(0,0,0,0.25)', 'important');
    btn.style.setProperty('opacity', '1', 'important');
    btn.style.setProperty('visibility', 'visible', 'important');
    btn.style.setProperty('pointer-events', 'auto', 'important');
  };

  const repairNative = () => {
    NATIVE_SELECTORS.forEach((selector) => {
      const target = doc.querySelector(selector);
      if (!target) return;
      target.style.setProperty('pointer-events', 'auto', 'important');
      target.style.setProperty('z-index', '100001', 'important');
    });
  };

  const tick = () => {
    repairNative();
    ensureCustomButton();
  };

  tick();
  window.setInterval(tick, 800);
})();
</script>
"""


def render_sidebar_toggle() -> None:
    """Repair stale hosted sidebar state and guarantee a working reopen control."""
    components.html(_TOGGLE_HTML, height=1, width=1)
