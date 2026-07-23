/**
 * 动画主控制器
 * @config target 动画对象，可以是数组，如果是数组的话会批量分发onframe等事件
 * @config life(1000) 动画时长
 * @config delay(0) 动画延迟时间
 * @config loop(true)
 * @config onframe
 * @config easing(optional)
 * @config ondestroy(optional)
 * @config onrestart(optional)
 *
 * TODO pause
 */

import easingFuncs, {AnimationEasing} from './easing';
import type Animation from './Animation';
import { isFunction, noop } from '../core/util';
import { createCubicEasingFunc } from './cubicEasing';

type OnframeCallback = (percent: number) => void;
type ondestroyCallback = () => void
type onrestartCallback = () => void

export type DeferredEventTypes = 'destroy' | 'restart'
// type DeferredEventKeys = 'ondestroy' | 'onrestart'

export interface ClipProps {
    life?: number
    delay?: number
    loop?: boolean
    easing?: AnimationEasing

    onframe?: OnframeCallback
    ondestroy?: ondestroyCallback
    onrestart?: onrestartCallback
}

export default class Clip {

    private _life: number
    private _delay: number

    private _inited: boolean = false
    private _startTime = 0 // 开始时间单位毫秒

    private _pausedTime = 0
    private _paused = false

    animation: Animation

    loop: boolean

    easing: AnimationEasing
    easingFunc: (p: number) => number

    // For linked list. Readonly
    next: Clip
    prev: Clip

    onframe: OnframeCallback
    ondestroy: ondestroyCallback
    onrestart: onrestartCallback

    constructor(opts: ClipProps) {

        this._life = opts.life || 1000;
        this._delay = opts.delay || 0;

        this.loop = opts.loop || false;

        this.onframe = opts.onframe || noop;
        this.ondestroy = opts.ondestroy || noop;
        this.onrestart = opts.onrestart || noop;

        opts.easing && this.setEasing(opts.easing);
    }

    step(globalTime: number, deltaTime: number): boolean {
        // Set startTime on first step, or _startTime may has milleseconds different between clips
        // PENDING
        if (!this._inited) {
            this._startTime = globalTime + this._delay;
            this._inited = true;
        }

        if (this._paused) {
            this._pausedTime += deltaTime;
            return;
        }

        const life = this._life;
        let elapsedTime = globalTime - this._startTime - this._pausedTime;
        let percent = elapsedTime / life;

        // PENDING: Not begin yet. Still run the loop.
        // In the case callback needs to be invoked.
        // Or want to update to the begin state at next frame when `setToFinal` and `delay` are both used.
        // To avoid the unexpected blink.
        if (percent < 0) {
            percent = 0;
        }

        percent = Math.min(percent, 1);

        const easingFunc = this.easingFunc;
        const schedule = easingFunc ? easingFunc(percent) : percent;

        this.onframe(schedule);

        // 结束
        if (percent === 1) {
            if (this.loop) {
                // Restart
                const remainder = elapsedTime % life;
                this._startTime = globalTime - remainder;
                this._pausedTime = 0;

                this.onrestart();
            }
            else {
                return true;
            }
        }

        return false;
    }

    pause() {
        this._paused = true;
    }

    resume() {
        this._paused = false;
    }

    setEasing(easing: AnimationEasing) {
        this.easing = easing;
        this.easingFunc = isFunction(easing)
            ? easing
            : easingFuncs[easing] || createCubicEasingFunc(easing);
    }
}