import Element, { ElementProps } from 'zrender/lib/Element.js';
import { ZREasing } from './types.js';
declare type AnimationWrapDoneCallback = () => void;
/**
 * Animate multiple elements with a single done-callback.
 *
 * @example
 *  animation
 *      .createWrap()
 *      .add(el1, {x: 10, y: 10})
 *      .add(el2, {shape: {width: 500}, style: {fill: 'red'}}, 400)
 *      .done(function () { // done })
 *      .start('cubicOut');
 */
declare class AnimationWrap {
    private _storage;
    private _elExistsMap;
    private _finishedCallback;
    /**
     * Caution: a el can only be added once, otherwise 'done'
     * might not be called. This method checks this (by el.id),
     * suppresses adding and returns false when existing el found.
     *
     * @return Whether adding succeeded.
     */
    add(el: Element, target: ElementProps, duration?: number, delay?: number, easing?: ZREasing): boolean;
    /**
     * Only execute when animation done/aborted.
     */
    finished(callback: AnimationWrapDoneCallback): AnimationWrap;
    /**
     * Will stop exist animation firstly.
     */
    start(): AnimationWrap;
}
export declare function createWrap(): AnimationWrap;
export {};
