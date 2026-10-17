:html_theme.sidebar_secondary.remove: true

Usage and Downloads
===================

.. raw:: html

   <div id="phgx-stats">
     <p style="color: var(--pst-color-text-muted, #6a7076); font-size: 0.85rem;">
       Loading usage statistics&hellip;
     </p>
   </div>

..
   Maintainer notes live with the code, not on this page:

   tools/fetch_usage_stats.py   data sources, the three installer buckets, and
                                the GoatCounter setup steps for the visitor map
   tools/preview_usage.py       serve this dashboard locally without a full
                                Sphinx build
   docs/source/_static/         usage-stats.js, usage-stats.css, and the
                                committed stats/ snapshot
   .github/workflows/stats.yml  the monthly refresh job
