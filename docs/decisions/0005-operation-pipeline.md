# 0005 — Operation pipeline

ArrayScope array-derived views are represented as a base data reference plus an
ordered list of immutable operations.

The operation pipeline lives in `arrayscope/operation_pipeline.py`. It must not
import Qt or pyqtgraph. GUI code may eventually translate user intent into
pipeline operations, but the operation definitions and evaluation rules remain
pure NumPy.

The first version supports materialized evaluation: applying the operation list
returns a concrete derived array. Each operation also exposes shape prediction so
the document can track `current_shape` without requiring GUI code to materialize
the array first. This keeps the API compatible with future lazy evaluation,
where operations could describe a derived view without immediately computing all
values.

Undo is operation-list based, not inverse-transform based. Removing the most
recent operation re-evaluates the remaining list from the untouched base data.
That is predictable for lossy operations such as mean or root-sum-squares, avoids
inventing invalid inverses, and keeps FFT/iFFT semantics explicit instead of
depending on reverse bookkeeping.

The base data reference is preserved by default. Operations derive views or
materialized arrays from it, but the document does not replace the base array
when an operation is appended. This makes undo a structural change to the
operation list rather than mutation of the source data.
