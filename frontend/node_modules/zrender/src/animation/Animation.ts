/**
 * Animation main class, dispatch and manage all animation controllers
 *
 */
// TODO Additive animation
// http://iosoteric.com/additive-animations-animatewithduration-in-ios-8/
// https://developer.apple.com/videos/wwdc2014/#236

import Eventful from '../core/Eventful';
import requestAnimationFrame from './requestAnimationFrame';
import Animator from './Animator';
import Clip from './Clip';

export function getTime() {
    return new Date().getTime();
}

interface Stage {
    update?: () => void
}

interface AnimationOption {
    stage?: Stage
}
/**
 * @example
 *     const animation = new Animation();
 *     const obj = {
 *         x: 100,
 *         y: 100
 *     };
 *     animation.animate(node.position)
 *         .when(1000, {
 *             x: 500,
 *             y: 500
 *         })
 *         .when(2000, {
 *             x: 100,
 *             y: 100
 *         })
 *         .start();
 */

export default class Animation extends Eventful {

    stage: Stage

    // Use linked list to store clip
    private _head: Clip
    private _tail: Clip

    private _running = false

    private _time = 0
    private _pausedTime = 0
    private _pauseStart = 0

    private _paused = false;

    constructor(opts?: AnimationOption) {
        super();

        opts = opts || {};

        this.stage = opts.stage || {};
    }

    /**
     * Add clip
     */
    addClip(clip: Clip) {
        if (clip.animation) {
            // Clip has been added
            this.removeClip(clip);
        }

        if (!this._head) {
            this._head = this._tail = clip;
        }
        else {
            this._tail.next = clip;
            clip.prev = this._tail;
            clip.next = null;
            this._tail = clip;
        }
        clip.animation = this;
    }
    /**
     * Add animator
     */
    addAnimator(animator: Animator<any>) {
        animator.animation = this;
        const clip = animator.getClip();
        if (clip) {
            this.addClip(clip);
        }
    }
    /**
     * Delete animation clip
     */
    removeClip(clip: Clip) {
        if (!clip.animation) {
            return;
        }
        const prev = clip.prev;
        const next = clip.next;
        if (prev) {
            prev.next = next;
        }
        else {
            // Is head
            this._head = next;
        }
        if (next) {
            next.prev = prev;
        }
        else {
            // Is tail
            this._tail = prev;
        }
        clip.next = clip.prev = clip.animation = null;
    }

    /**
     * Delete animation clip
     */
    removeAnimator(animator: Animator<any>) {
        const clip = animator.getClip();
        if (clip) {
            this.removeClip(clip);
        }
        animator.animation = null;
    }

    update(notTriggerFrameAndStageUpdate?: boolean) {
        const time = getTime() - this._pausedTime;
        const delta = time - this._time;
        let clip = this._head;

        while (clip) {
            // Save the nextClip before step.
            // So the loop will not been affected if the clip is removed in the callback
            const nextClip = clip.next;
            let finished = clip.step(time, delta);
            if (finished) {
                clip.ondestroy();
                this.removeClip(clip);
                clip = nextClip;
            }
            else {
                clip = nextClip;
            }
        }

        this._time = time;

        if (!notTriggerFrameAndStageUpdate) {

            // 'frame' should be triggered before stage, because upper application
            // depends on the sequence (e.g., echarts-stream and finish
            // event judge)
            this.trigger('frame', delta);

            this.stage.update && this.stage.update();
        }
    }

    _startLoop() {
        const self = this;

        this._running = true;

        function step() {
            if (self._running) {
                requestAnimationFrame(step);
                !self._paused && self.update();
            }
        }

        requestAnimationFrame(step);
    }

    /**
     * Start animation.
     */
    start() {
        if (this._running) {
            return;
        }

        this._time = getTime();
        this._pausedTime = 0;

        this._startLoop();
    }

    /**
     * Stop animation.
     */
    stop() {
        this._running = false;
    }

    /**
     * Pause animation.
     */
    pause() {
        if (!this._paused) {
            this._pauseStart = getTime();
            this._paused = true;
        }
    }

    /**
     * Resume animation.
     */
    resume() {
        if (this._paused) {
            this._pausedTime += getTime() - this._pauseStart;
            this._paused = false;
        }
    }

    /**
     * Clear animation.
     */
    clear() {
        let clip = this._head;

        while (clip) {
            let nextClip = clip.next;
            clip.prev = clip.next = clip.animation = null;
            clip = nextClip;
        }

        this._head = this._tail = null;
    }

    /**
     * Whether animation finished.
     */
    isFinished() {
        return this._head == null;
    }

    /**
     * Creat animator for a target, whose props can be animated.
     */
    // TODO Gap
    animate<T>(target: T, options: {
        loop?: boolean  // Whether loop animation
    }) {
        options = options || {};

        // Start animation loop
        this.start();

        const animator = new Animator(
            target,
            options.loop
        );

        this.addAnimator(animator);

        return animator;
    }
}