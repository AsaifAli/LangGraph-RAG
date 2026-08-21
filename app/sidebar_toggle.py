"""Reliable Streamlit sidebar interaction repair.

The primary sidebar is still Streamlit's native control.  This tiny browser-side
bridge only repairs pointer-events/z-index when Streamlit has rendered the
native control but the surrounding layout layer has made it difficult to click.
"""
from __future__ import annotations

import streamlit.components.v1 as components


def render_sidebar_toggle() -> None:
    """Keep Streamlit's native sidebar controls clickable after collapse."""
    components.html(
        """
        <script>
          (() => {
            const doc = window.parent.document;
            const selectors = [
              '[data-testid="stSidebarCollapsedControl"] button',
              '[data-testid="stSidebarCollapsedControl"]',
              '[data-testid="stSidebarCollapseButton"] button',
              '[data-testid="stSidebarCollapseButton"]',
              '[data-testid="collapsedControl"] button',
              '[data-testid="collapsedControl"]'
            ];
            const repair = () => {
              for (const selector of selectors) {
                const target = doc.querySelector(selector);
                if (target) {
                  target.style.pointerEvents = 'auto';
                  target.style.zIndex = '100001';
                }
              }
            };
            repair();
            window.setInterval(repair, 1200);
          })();
        </script>
        """,
        height=1,
        width=1,
    )
