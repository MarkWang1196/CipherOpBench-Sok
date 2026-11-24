// Timeline controller (spec §16.3). Owns playback state only — rendering and
// animation are injected, so the controller is testable under `node --test`
// with stubs and never touches the DOM directly.
//
// Invariants:
//   * at most one transition in flight; a new seek cancels the old one
//     (token-based superseding).
//   * play stops automatically at the last step.
//   * pausing or destroying cancels the pending timer.
//   * reduced motion keeps step *changes* but collapses durations.

const BASE_STEP_MS = 1500;

export class TimelineController {
  /**
   * @param {object} trace validated trace
   * @param {{renderStep:(trace, index)=>void}} renderer
   * @param {{animate:(opts:object)=>Promise, cancel?:()=>void}|null} animator
   * @param {object} [options]
   * @param {boolean} [options.reducedMotion]
   * @param {(controller)=>void} [options.onStateChange]
   */
  constructor(trace, renderer, animator, options = {}) {
    this.trace = trace;
    this.renderer = renderer;
    this.animator = animator || null;
    this._reducedMotion = Boolean(options.reducedMotion);
    this._onStateChange = typeof options.onStateChange === "function" ? options.onStateChange : null;
    this._current = 0;
    this._playing = false;
    this._speed = 1;
    this._timer = null;
    this._transitionId = 0;
    this._destroyed = false;
    // Render the first step immediately.
    if (this.trace && this.trace.steps && this.trace.steps.length) {
      this.renderer.renderStep(this.trace, 0);
    }
  }

  get currentStep() {
    return this._current;
  }

  get isPlaying() {
    return this._playing;
  }

  get speed() {
    return this._speed;
  }

  get totalSteps() {
    return this.trace.steps.length;
  }

  _emit() {
    if (this._onStateChange) this._onStateChange(this);
  }

  async _transition(to, opts = {}) {
    const total = this.trace.steps.length;
    const target = Math.max(0, Math.min(total - 1, to));
    const id = ++this._transitionId;

    if (this._timer) {
      clearTimeout(this._timer);
      this._timer = null;
    }

    if (target === this._current && !opts.replay) {
      this.renderer.renderStep(this.trace, target);
      this._emit();
      return;
    }

    if (this.animator) this.animator.cancel?.();

    const from = this._current;
    this._current = target;
    this._emit();

    if (this.animator) {
      await this.animator.animate({
        trace: this.trace,
        from,
        to: target,
        speed: this._speed,
        reducedMotion: this._reducedMotion,
      });
    } else {
      this.renderer.renderStep(this.trace, target);
    }

    // A newer transition superseded us (rapid seek) or we were destroyed.
    if (this._destroyed || id !== this._transitionId) return;
    this._emit();
  }

  async next() {
    await this._transition(this._current + 1);
  }

  async previous() {
    await this._transition(this._current - 1);
  }

  async seek(index) {
    await this._transition(index);
  }

  async replay() {
    await this._transition(this._current, { replay: true });
  }

  play() {
    if (this._playing || this._destroyed) return;
    this._playing = true;
    this._emit();
    this._scheduleNext();
  }

  pause() {
    this._playing = false;
    if (this._timer) {
      clearTimeout(this._timer);
      this._timer = null;
    }
    this._emit();
  }

  _stepDelay(step) {
    if (this._reducedMotion) return 40;
    const base = step && step.durationMs ? step.durationMs : BASE_STEP_MS;
    return Math.max(80, base / this._speed);
  }

  _scheduleNext() {
    if (this._destroyed || !this._playing) return;
    if (this._current >= this.totalSteps - 1) {
      this.pause(); // auto-stop at the end
      return;
    }
    const step = this.trace.steps[this._current];
    this._timer = setTimeout(async () => {
      this._timer = null;
      if (!this._playing || this._destroyed) return;
      await this._transition(this._current + 1);
      if (this._playing && !this._destroyed) this._scheduleNext();
    }, this._stepDelay(step));
  }

  setSpeed(multiplier) {
    this._speed = multiplier;
    if (this._playing) {
      if (this._timer) {
        clearTimeout(this._timer);
        this._timer = null;
      }
      this._scheduleNext();
    }
    this._emit();
  }

  destroy() {
    this._destroyed = true;
    this.pause();
    this._transitionId += 1;
    if (this.animator) this.animator.cancel?.();
    this._onStateChange = null;
  }
}
