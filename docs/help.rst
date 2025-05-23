.. include:: links.inc

.. _faq:

================================
Frequently Asked Questions (FAQ)
================================


How do I cite Braindecode?
--------------------------
See :doc:`cite`.

Help! I can't get Python and Braindecode working!
-------------------------------------------------

Check out if your are running the most recent version of braindecode.


I still can't get it to work!
-----------------------------

When you encounter an error message or unexpected results, it can be hard to
tell whether it happened because of a bug in Braindecode, a mistake in user
code, a corrupted data file, or irregularities in the data itself.

Your first step when asking for help should be the `Braindecode Chat
<braindecode-chat_>`_, not GitHub. This bears
repeating: *the GitHub issue tracker is not for usage help* — it is for
software bugs, feature requests, and improvements to documentation. If you
open an issue that contains only a usage question, we will close the issue and
direct you to the chat.

I think I found a bug, what do I do?
------------------------------------

If you're pretty sure the problem you've encountered
is a software bug (not bad data or user error):

- Make sure you're using the most current version. You can check it locally
  at a shell prompt with:

  .. code-block:: console

      $ python -c "import braindecode; print(braindecode.__version__)"

  which will also give you version info about braindecode.

- If you're already on the most current version, if possible try using
  :ref:`the latest development version <install_source>`, as the bug may
  have been fixed already since the latest release. If you can't try the latest
  development version, search the GitHub issues page to see if the problem has
  already been reported and/or fixed.

If the problem persists, `open a new issue <braindecode-issues_>`_
and include the *smallest possible* code sample that replicates the error
you're seeing. Paste the code sample into the issue, with a line containing
three backticks (\`\`\`) above and below the lines of code. This
`minimal working example <https://en.wikipedia.org/wiki/Minimal_Working_Example>`__
should be self-contained, which means that
Braindecode contributors should be able to copy and paste the provided snippet
and replicate the bug on their own computers.
