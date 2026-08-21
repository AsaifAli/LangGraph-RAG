"""Browser-side repair for Streamlit's persisted sidebar state.

EvidenceFlow uses Streamlit's native sidebar.  The browser bridge has two small
jobs:

1. On the first visit in a browser tab, clear stale sidebar layout preferences
   that can leave a hosted/fullscreen Streamlit app collapsed.  The page is
   reloaded once so ``initial_sidebar_state=\"expanded\"`` can take effect.
2. Keep Streamlit's native collapse/re-open controls clickable when framework
   chrome or a stale layout layer interferes with pointer events.

This intentionally does not create a replacement sidebar or fake toggle.
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

  const selectors = [
    '[data-testid="stSidebarCollapsedControl"] button',
    '[data-testid="stSidebarCollapsedControl"]',
    '[data-testid="stSidebarCollapseButton"] button',
    '[data-testid="stSidebarCollapseButton"]',
    '[data-testid="collapsedControl"] button',
    '[data-testid="collapsedControl"]'
  ];

  const repair = () => {
    selectors.forEach((selector) => {
      const target = doc.querySelector(selector);
      if (!target) return;
      target.style.pointerEvents = 'auto';
      target.style.zIndex = '100001';
      target.style.visibility = 'visible';
      target.style.opacity = '1';
    });
  };

  repair();
  window.setInterval(repair, 1200);
})();
</script>
"""


def render_sidebar_toggle() -> None:
    """Repair stale hosted sidebar state and keep native controls clickable."""
    components.html(_TOGGLE_HTML, height=1, width=1)
