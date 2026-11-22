.. _install_gaudi:

########################
 Running on Intel Gaudi
########################

Braindecode runs on Intel Gaudi accelerators (the ``hpu`` device) with PyTorch from
Intel's Gaudi software stack. Gaps of that stack are filled outside braindecode: by two
plugins, and by one setting of Habana's own.

*********
 Install
*********

Install PyTorch from the Gaudi software stack first: a ``torch`` wheel from PyPI would
replace Intel's build. Then install braindecode and the two plugins:

.. code-block:: bash

    pip install braindecode
    pip install git+https://github.com/bruAristimunha/hpu-complex
    pip install git+https://github.com/bruAristimunha/hpu-ops

- `hpu-complex <https://github.com/bruAristimunha/hpu-complex>`_ adds complex tensors,
  ``torch.fft`` and ``torch.stft`` on HPU.
- `hpu-ops <https://github.com/bruAristimunha/hpu-ops>`_ fixes kernels and compile
  passes that Gaudi lacks or gets wrong.

Enable both before building or compiling a model:

.. code-block:: python

    import hpu_complex
    import hpu_ops

    hpu_complex.enable()
    hpu_ops.enable()

***********
 Attention
***********

Braindecode calls torch's :func:`torch.nn.functional.scaled_dot_product_attention` on
every device. On Gaudi it runs op by op, through the full score matrix, unless Habana's
attention setting is on. With it, the bridge's fused kernel runs every call, with or
without a mask, eager and compiled. The setting depends on the bridge version. Export
the variables before Python starts, because Habana's library reads them when it loads:

.. list-table::
    :header-rows: 1

    - - Gaudi software
      - Attention setting
    - - 1.21
      - ``export PT_HPU_USE_OVERRIDE_ATEN_SDPA=1``
    - - 1.22
      - ``export PT_HPU_USE_ATEN_SDPA=0``
    - - 1.23 and later, with torch >= 2.8
      - ``torch.nn.attention.sdpa_kernel([SDPBackend.OVERRIDEABLE])`` around the forward
        and backward passes

For 1.23 and later:

.. code-block:: python

    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel([SDPBackend.OVERRIDEABLE]):
        loss = criterion(model(X), y)
        loss.backward()

****************
 Execution mode
****************

Eager (``PT_HPU_LAZY_MODE=0``) or ``torch.compile(backend="hpu_backend")``; lazy mode is
not supported.

When compiling for training or evaluation:

- Pass ``dynamic=False`` to ``torch.compile(model, backend="hpu_backend",
  dynamic=False)`` so the compiler specializes on fixed input shapes instead of tracing
  symbolic shape guards.
- Pass ``drop_last=True`` to :class:`torch.utils.data.DataLoader` during training so a
  shorter final batch does not trigger a second graph compilation or hit dynamic-batch
  kernel bugs.
- Prefer out-of-place activations such as ``nn.ReLU()`` over ``nn.ReLU(inplace=True)``
  in custom modules and heads, because in-place mutations on view outputs can abort
  graph capture and autograd version checks.

**************
 Known issues
**************

Measured with Gaudi software 1.21.4 and the attention setting on.

Attention on part of a larger tensor in eager mode
    In eager mode, attention fails with ``synLaunch failed ... synStatus 30 [Sections'
    validation failure]`` when its query, key and value are views of part of a larger
    tensor, e.g. a batch or token slice taken after the projection, and an earlier call
    took views of a tensor of the same size. Compiled models are not affected, and
    neither are copies (``.clone()``) of the slice. LaBraM, and every model that calls
    :func:`braindecode.functional.scaled_dot_product_attention`, take q, k and v from a
    projection of the current batch, so their views span the whole projection.

Same-size attention calls in eager mode
    In eager mode, attention can return wrong values, without an error, when its query,
    key and value are transposed views (``x.view(n, L, H, D).transpose(1, 2)``) and an
    earlier call took views with as many elements in another shape, such as axial
    attention over the time and then the channel axis of one tensor.
    :class:`braindecode.models.AXON` calls ``.contiguous()`` on query, key and value
    before attention so eager execution stays accurate; for custom axial modules, either
    materialize contiguous head tensors or set ``PT_HPU_ENABLE_EAGER_JIT_CACHE=0``.
    Compiled models are not affected.

Learnable attention masks
    The fused kernel computes no gradient for ``attn_mask``. A floating-point mask that
    requires one, like the learnable channel bias of :class:`braindecode.models.DIVER1`,
    gets none with 1.21; according to the bridge's source, 1.22 returns the mask itself
    as its gradient and 1.24 none. Train such models with the attention setting off:
    attention then runs op by op and computes it.

Fully masked query rows
    When a query has no key to attend to, the fused kernel returns NaN with an additive
    ``-inf`` mask, and finite, non-zero values with a boolean mask; torch returns zeros.
    In braindecode's models, a ``ch_mask`` with at least one real channel per sample
    never leaves a query without a key. :func:`braindecode.functional.check_ch_mask`
    requires such a mask, but checks the values in eager mode only.
