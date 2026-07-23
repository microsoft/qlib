import easingFuncs from './easing.js';
import { isFunction, noop } from '../core/util.js';
import { createCubicEasingFunc } from './cubicEasing.js';
var Clip = (function () {
    function Clip(opts) {
        this._inited = false;
        this._startTime = 0;
        this._pausedTime = 0;
        this._paused = false;
        this._life = opts.life || 1000;
        this._delay = opts.delay || 0;
        this.loop = opts.loop || false;
        this.onframe = opts.onframe || noop;
        this.ondestroy = opts.ondestroy || noop;
        this.onrestart = opts.onrestart || noop;
        opts.easing && this.setEasing(opts.easing);
    }
    Clip.prototype.step = function (globalTime, deltaTime) {
        if (!this._inited) {
            this._startTime = globalTime + this._delay;
            this._inited = true;
        }
        if (this._paused) {
            this._pausedTime += deltaTime;
            return;
        }
        var life = this._life;
        var elapsedTime = globalTime - this._startTime - this._pausedTime;
        var percent = elapsedTime / life;
        if (percent < 0) {
            percent = 0;
        }
        percent = Math.min(percent, 1);
        var easingFunc = this.easingFunc;
        var schedule = easingFunc ? easingFunc(percent) : percent;
        this.onframe(schedule);
        if (percent === 1) {
            if (this.loop) {
                var remainder = elapsedTime % life;
                this._startTime = globalTime - remainder;
                this._pausedTime = 0;
                this.onrestart();
            }
            else {
                return true;
            }
        }
        return false;
    };
    Clip.prototype.pause = function () {
        this._paused = true;
    };
    Clip.prototype.resume = function () {
        this._paused = false;
    };
    Clip.prototype.setEasing = function (easing) {
        this.easing = easing;
        this.easingFunc = isFunction(easing)
            ? easing
            : easingFuncs[easing] || createCubicEasingFunc(easing);
    };
    return Clip;
}());
export default Clip;
