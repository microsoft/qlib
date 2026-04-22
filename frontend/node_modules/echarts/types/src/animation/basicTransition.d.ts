import { AnimationOptionMixin, AnimationOption } from '../util/types.js';
import Element, { ElementAnimateConfig, ElementProps } from 'zrender/lib/Element.js';
import Model from '../model/Model.js';
import Displayable from 'zrender/lib/graphic/Displayable.js';
export declare const transitionStore: (hostObj: Displayable<import("zrender/lib/graphic/Displayable").DisplayableProps>) => {
    oldStyle: Displayable['style'];
};
declare type AnimateOrSetPropsOption = {
    dataIndex?: number;
    cb?: () => void;
    during?: (percent: number) => void;
    removeOpt?: AnimationOption;
    isFrom?: boolean;
};
/**
 * Return null if animation is disabled.
 */
export declare function getAnimationConfig(animationType: 'enter' | 'update' | 'leave', animatableModel: Model<AnimationOptionMixin>, dataIndex: number, extraOpts?: Pick<ElementAnimateConfig, 'easing' | 'duration' | 'delay'>, extraDelayParams?: unknown): Pick<ElementAnimateConfig, 'easing' | 'duration' | 'delay'> | null;
/**
 * Update graphic element properties with or without animation according to the
 * configuration in series.
 *
 * Caution: this method will stop previous animation.
 * So do not use this method to one element twice before
 * animation starts, unless you know what you are doing.
 * @example
 *     graphic.updateProps(el, {
 *         position: [100, 100]
 *     }, seriesModel, dataIndex, function () { console.log('Animation done!'); });
 *     // Or
 *     graphic.updateProps(el, {
 *         position: [100, 100]
 *     }, seriesModel, function () { console.log('Animation done!'); });
 */
declare function updateProps<Props extends ElementProps>(el: Element<Props>, props: Props, animatableModel?: Model<AnimationOptionMixin>, dataIndex?: AnimateOrSetPropsOption['dataIndex'] | AnimateOrSetPropsOption['cb'] | AnimateOrSetPropsOption, cb?: AnimateOrSetPropsOption['cb'] | AnimateOrSetPropsOption['during'], during?: AnimateOrSetPropsOption['during']): void;
export { updateProps };
/**
 * Init graphic element properties with or without animation according to the
 * configuration in series.
 *
 * Caution: this method will stop previous animation.
 * So do not use this method to one element twice before
 * animation starts, unless you know what you are doing.
 */
export declare function initProps<Props extends ElementProps>(el: Element<Props>, props: Props, animatableModel?: Model<AnimationOptionMixin>, dataIndex?: AnimateOrSetPropsOption['dataIndex'] | AnimateOrSetPropsOption['cb'] | AnimateOrSetPropsOption, cb?: AnimateOrSetPropsOption['cb'] | AnimateOrSetPropsOption['during'], during?: AnimateOrSetPropsOption['during']): void;
/**
 * If element is removed.
 * It can determine if element is having remove animation.
 */
export declare function isElementRemoved(el: Element): boolean;
/**
 * Remove graphic element
 */
export declare function removeElement<Props>(el: Element<Props>, props: Props, animatableModel?: Model<AnimationOptionMixin>, dataIndex?: AnimateOrSetPropsOption['dataIndex'] | AnimateOrSetPropsOption['cb'] | AnimateOrSetPropsOption, cb?: AnimateOrSetPropsOption['cb'] | AnimateOrSetPropsOption['during'], during?: AnimateOrSetPropsOption['during']): void;
export declare function removeElementWithFadeOut(el: Element, animatableModel?: Model<AnimationOptionMixin>, dataIndex?: number): void;
/**
 * Save old style for style transition in universalTransition module.
 * It's used when element will be reused in each render.
 * For chart like map, heatmap, which will always create new element.
 * We don't need to save this because universalTransition can get old style from the old element
 */
export declare function saveOldStyle(el: Displayable): void;
export declare function getOldStyle(el: Displayable): import("zrender/lib/core/types").Dictionary<any>;
