.. The release notes live in CHANGELOG.md at the repository root, so there is
   one copy to edit. conf.py does not load myst_parser as a Sphinx extension,
   so the file is read with MyST's docutils parser, which needs only the
   myst-parser package from docs/requirements.txt. If myst_parser is added to
   the extensions in conf.py, switch the parser below to myst_parser.sphinx_.

.. include:: ../../CHANGELOG.md
   :parser: myst_parser.docutils_
