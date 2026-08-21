from __future__ import annotations

import streamlit as st


_TOGGLE_HTML = r'''
<div class="ef-sidebar-fallback" aria-hidden="true">
  <button id="ef-sidebar-open" type="button" aria-label="Open navigation" title="Open navigation">
    <span aria-hidden="true">☰</span>
  </button>
</div>
<script>
(() => {
  const root = document.currentScript?.previousElementSibling;
  const button = root?.querySelector('#ef-sidebar-open');
  if (!button) return;

  const parentDoc = () => window.parent && window.parent.document;

  function sidebar() {
    return parentDoc()?.querySelector('section[data-testid="stSidebar"]');
  }

  function nativeToggle() {
    const doc = parentDoc();
    if (!doc) return;
    const selectors = [
      '[data-testid="stSidebarCollapseButton"] button',
      '[data-testid="stSidebarCollapseButton"] [data-testid="baseButton-headerNoPadding"]',
      '[data-testid="stSidebarCollapsedControl"] button',
      '[data-testid="collapsedControl"] button'
    ];
    for (const selector of selectors) {
      const target = doc.querySelector(selector);
      if (target) {
        target.click();
        return;
      }
    }
  }

  function sync() {
    const expanded = sidebar()?.getAttribute('aria-expanded') === 'true';
    button.style.display = expanded ? 'none' : 'inline-flex';
    root?.setAttribute('aria-hidden', expanded ? 'true' : 'false');
  }

  button.addEventListener('click', nativeToggle);
  sync();

  const observer = new MutationObserver(sync);
  const doc = parentDoc();
  if (doc?.body) {
    observer.observe(doc.body, {
      subtree: true,
      attributes: true,
      attributeFilter: ['aria-expanded']
    });
  }
})();
</script>
'''


def render_sidebar_toggle() -> None:
    """Provide a reliable reopen affordance for Streamlit's collapsed sidebar.

    Streamlit's native collapsed control can remain visibility-hidden on some
    versions.  The trusted iframe runs same-origin JS and programmatically
    clicks Streamlit's real React control, while disappearing whenever the
    sidebar is expanded.
    """
    st.iframe(_TOGGLE_HTML, width=1, height=1, scrolling=False, tab_index=-1)
