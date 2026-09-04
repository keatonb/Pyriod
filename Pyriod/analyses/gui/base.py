"""Base classes for optional Pyriod analysis GUI tabs.

This module defines the interface shared by GUI representations of optional
analysis routines.  Scientific calculations should remain in headless modules
under :mod:`Pyriod.analyses`; subclasses of :class:`AnalysisGUI` should only
manage widgets, plotting, user interaction, and presentation of those results.
"""

from __future__ import annotations

import asyncio
from abc import ABC
from collections.abc import Callable, Coroutine
from threading import Event
from typing import Any

import ipywidgets as widgets
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class AnalysisGUI(ABC):
    """Base class for optional Pyriod analysis GUI tabs.

    `AnalysisGUI` provides the small amount of infrastructure shared by
    specialized analysis interfaces while leaving their scientific
    calculations and layouts independent.  Subclasses should construct their
    top-level widget, assign it with :meth:`_set_widget`, and use headless
    functions from :mod:`Pyriod.analyses` for numerical work.

    Parameters
    ----------
    pw : Prewhitener
        Analysis object whose data and settings are used by the GUI.
    log_updated : callable or None, optional
        Callback invoked after this GUI writes a message to the `Prewhitener`
        log.  A `PyriodGUI` can pass its log-refresh method here.
    busy_changed : callable or None, optional
        Callback invoked with a single boolean argument whenever the busy
        state changes.  A parent `PyriodGUI` can use this to synchronize
        controls or status displays.

    Attributes
    ----------
    pw : Prewhitener
        The associated `Prewhitener`.
    title : str
        Label used for the analysis tab.

    Notes
    -----
    Subclasses should normally:

    1. call ``super().__init__(...)``;
    2. construct widgets and figures;
    3. register widget callbacks with :meth:`_on_click` or :meth:`_observe`;
    4. register figures with :meth:`_track_figure`;
    5. call :meth:`_set_widget` with the top-level widget.

    The base :meth:`close` method then disconnects registered callbacks,
    requests cancellation of active work, closes tracked figures, and closes
    the top-level widget.
    """

    title = "Analysis"

    def __init__(
        self,
        pw,
        *,
        log_updated: Callable[[], None] | None = None,
        busy_changed: Callable[[bool], None] | None = None,
    ):
        self.pw = pw

        self._log_updated = log_updated
        self._busy_changed = busy_changed

        self._widget: widgets.Widget | None = None
        self._closed = False
        self._busy = False

        self._cancel_event = Event()
        self._task: asyncio.Task | None = None

        self._cleanup_callbacks: list[Callable[[], None]] = []
        self._figures: list[Figure] = []

    @property
    def widget(self) -> widgets.Widget:
        """Top-level widget displayed in the Pyriod tab.

        Returns
        -------
        ipywidgets.Widget
            Top-level widget for this analysis interface.

        Raises
        ------
        RuntimeError
            If the subclass has not yet assigned its top-level widget with
            :meth:`_set_widget`.
        """
        if self._widget is None:
            raise RuntimeError(
                f"{type(self).__name__} has not initialized its top-level widget."
            )
        return self._widget

    @property
    def closed(self) -> bool:
        """Whether this analysis GUI has been closed."""
        return self._closed

    @property
    def busy(self) -> bool:
        """Whether this analysis GUI currently has active work."""
        return self._busy

    @property
    def cancel_requested(self) -> bool:
        """Whether cancellation has been requested."""
        return self._cancel_event.is_set()

    def _set_widget(self, widget: widgets.Widget) -> None:
        """Set the top-level widget returned to the Pyriod tab manager."""
        self._widget = widget

    def _log(self, message: str, *, level: str = "info") -> None:
        """Write a message to the associated `Prewhitener` log.

        Parameters
        ----------
        message : str
            Message to record.
        level : str, optional
            Log level passed to ``pw.log``. Default is ``"info"``.
        """
        self.pw.log(message, level=level)

        if self._log_updated is not None:
            self._log_updated()

    def _set_busy(self, busy: bool) -> None:
        """Update this interface's busy state and notify its parent."""
        busy = bool(busy)

        if busy == self._busy:
            return

        self._busy = busy

        if self._busy_changed is not None:
            self._busy_changed(busy)

    def _clear_cancel(self) -> None:
        """Clear a previous cancellation request before starting new work."""
        self._cancel_event.clear()

    def _request_cancel(self) -> None:
        """Request cooperative cancellation of active analysis work."""
        self._cancel_event.set()

    def _cancel_check(self) -> bool:
        """Return True when active work should stop."""
        return self._closed or self._cancel_event.is_set()

    def _log_exception(
        self,
        exc: BaseException,
        *,
        context: str = "Analysis failed",
    ) -> None:
        """Write an analysis exception to the `Prewhitener` log.

        Parameters
        ----------
        exc : BaseException
            Exception raised while running an analysis.
        context : str, optional
            Short description of the operation that failed. Default is
            ``"Analysis failed"``.

        Notes
        -----
        Normal asyncio cancellation is not logged as an analysis error.
        """
        if isinstance(exc, asyncio.CancelledError):
            return

        message = f"{context}: {type(exc).__name__}: {exc}"

        try:
            self._log(message, level="error")
        except Exception:
            # Never let failure of the logging path mask the original error.
            pass

    def _run_analysis(
        self,
        func: Callable,
        /,
        *args,
        context: str | None = None,
        **kwargs,
    ):
        """Run a synchronous analysis function and redirect errors to the log.

        Parameters
        ----------
        func : callable
            Function to execute.
        *args
            Positional arguments passed to `func`.
        context : str or None, optional
            Description used in the log entry if the function raises. If None,
            the function name is used.
        **kwargs
            Keyword arguments passed to `func`.

        Returns
        -------
        object or None
            Return value from `func`, or None if the function raises.
        """
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            if context is None:
                name = getattr(func, "__name__", type(func).__name__)
                context = f"{name} failed"
            self._log_exception(exc, context=context)
            return None

    async def _run_analysis_in_thread(
        self,
        func: Callable,
        /,
        *args,
        context: str | None = None,
        **kwargs,
    ):
        """Run a blocking analysis in a worker thread and log errors.

        Parameters
        ----------
        func : callable
            Blocking analysis function to execute.
        *args
            Positional arguments passed to `func`.
        context : str or None, optional
            Description used in the log entry if the function raises. If None,
            the function name is used.
        **kwargs
            Keyword arguments passed to `func`.

        Returns
        -------
        object or None
            Return value from `func`, or None if the function raises.

        Notes
        -----
        `asyncio.CancelledError` is re-raised so task cancellation retains its
        normal semantics. Other exceptions are redirected to the Pyriod log.
        """
        try:
            return await asyncio.to_thread(func, *args, **kwargs)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if context is None:
                name = getattr(func, "__name__", type(func).__name__)
                context = f"{name} failed"
            self._log_exception(exc, context=context)
            return None

    def _start_task(self, coroutine):
        """Start and track one asynchronous GUI task."""

        if self._closed:
            raise RuntimeError(
                "Cannot start work on a closed analysis GUI."
            )

        if self._task is not None and not self._task.done():
            raise RuntimeError(
                "An analysis task is already running."
            )

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()

        task = loop.create_task(coroutine)
        self._task = task

        def _clear_finished_task(finished):
            if self._task is finished:
                self._task = None

            try:
                finished.result()
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                self._log_exception(
                    exc,
                    context="Analysis task failed",
                )

        task.add_done_callback(_clear_finished_task)

        return task

    async def _to_thread(self, func: Callable, /, *args, **kwargs):
        """Run a blocking function in a worker thread without catching errors.

        For normal analysis execution, prefer :meth:`_run_analysis_in_thread`,
        which redirects analysis exceptions to the `Prewhitener` log.
        """
        return await asyncio.to_thread(func, *args, **kwargs)

    def _on_click(
        self,
        button: widgets.Button,
        handler: Callable,
    ) -> None:
        """Register a button callback that is disconnected on :meth:`close`."""
        button.on_click(handler)

        def cleanup() -> None:
            button.on_click(handler, remove=True)

        self._cleanup_callbacks.append(cleanup)

    def _observe(
        self,
        widget: widgets.Widget,
        handler: Callable,
        *,
        names: str | list[str] = "value",
    ) -> None:
        """Register a trait observer that is disconnected on :meth:`close`."""
        widget.observe(handler, names=names)

        def cleanup() -> None:
            widget.unobserve(handler, names=names)

        self._cleanup_callbacks.append(cleanup)

    def _track_figure(self, fig: Figure) -> Figure:
        """Register a Matplotlib figure to be closed with this interface."""
        self._figures.append(fig)
        return fig

    def _add_cleanup(self, callback: Callable[[], None]) -> None:
        """Register an arbitrary cleanup callback.

        Callbacks are executed in reverse registration order by :meth:`close`.
        """
        self._cleanup_callbacks.append(callback)

    def refresh(self) -> None:
        """Synchronize inexpensive display state with the `Prewhitener`.

        Subclasses may override this method when their display should react to
        changes in the underlying `Prewhitener`.  Expensive scientific
        calculations should not normally be triggered here.
        """
        return None

    def close(self) -> None:
        """Close the analysis GUI and release resources it owns.

        This method is idempotent.  It requests cooperative cancellation,
        disconnects registered callbacks, cancels the tracked asyncio task,
        closes tracked Matplotlib figures, closes the top-level widget, and
        clears the busy state.
        """
        if self._closed:
            return

        self._closed = True
        self._request_cancel()

        task = self._task
        if task is not None and not task.done():
            task.cancel()
        self._task = None

        for cleanup in reversed(self._cleanup_callbacks):
            try:
                cleanup()
            except Exception:
                pass
        self._cleanup_callbacks.clear()

        for fig in self._figures:
            try:
                plt.close(fig)
            except Exception:
                pass
        self._figures.clear()

        if self._widget is not None:
            try:
                self._widget.close()
            except Exception:
                pass
            self._widget = None

        # Notify the parent if this GUI is closed while marked busy.
        if self._busy:
            self._busy = False
            if self._busy_changed is not None:
                try:
                    self._busy_changed(False)
                except Exception:
                    pass
