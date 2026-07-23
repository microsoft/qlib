import { AnimationEasing } from 'zrender/lib/animation/easing.js';
import Element from 'zrender/lib/Element.js';
import { AnimationOption, AnimationOptionMixin } from '../util/types.js';
import type Model from '../model/Model.js';
declare type AnimationKeyframe<T extends Record<string, any>> = T & {
    easing?: AnimationEasing;
    percent?: number;
};
export interface ElementKeyframeAnimationOption<Props extends Record<string, any>> extends AnimationOption {
    loop?: boolean;
    keyframes?: AnimationKeyframe<Props>[];
}
/**
 * Stop previous keyframe animation and restore the attributes.
 * Avoid new keyframe animation starts with wrong internal state when the percent: 0 is not set.
 */
export declare function stopPreviousKeyframeAnimationAndRestore(el: Element): void;
export declare function applyKeyframeAnimation<T extends Record<string, any>>(el: Element, animationOpts: ElementKeyframeAnimationOption<T> | ElementKeyframeAnimationOption<T>[], animatableModel: Model<AnimationOptionMixin>): void;
export {};
