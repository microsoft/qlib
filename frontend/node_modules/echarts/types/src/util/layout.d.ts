import BoundingRect from 'zrender/lib/core/BoundingRect.js';
import { BoxLayoutOptionMixin, ComponentLayoutMode } from './types.js';
import Group from 'zrender/lib/graphic/Group.js';
import Element from 'zrender/lib/Element.js';
export interface LayoutRect extends BoundingRect {
    margin: number[];
}
export interface NewlineElement extends Element {
    newline: boolean;
}
/**
 * @public
 */
export declare const LOCATION_PARAMS: readonly ["left", "right", "top", "bottom", "width", "height"];
/**
 * @public
 */
export declare const HV_NAMES: readonly [readonly ["width", "left", "right"], readonly ["height", "top", "bottom"]];
declare function boxLayout(orient: 'horizontal' | 'vertical', group: Group, gap: number, maxWidth?: number, maxHeight?: number): void;
/**
 * VBox or HBox layouting
 * @param {string} orient
 * @param {module:zrender/graphic/Group} group
 * @param {number} gap
 * @param {number} [width=Infinity]
 * @param {number} [height=Infinity]
 */
export declare const box: typeof boxLayout;
/**
 * VBox layouting
 * @param {module:zrender/graphic/Group} group
 * @param {number} gap
 * @param {number} [width=Infinity]
 * @param {number} [height=Infinity]
 */
export declare const vbox: (group: Group, gap: number, maxWidth?: number, maxHeight?: number) => void;
/**
 * HBox layouting
 * @param {module:zrender/graphic/Group} group
 * @param {number} gap
 * @param {number} [width=Infinity]
 * @param {number} [height=Infinity]
 */
export declare const hbox: (group: Group, gap: number, maxWidth?: number, maxHeight?: number) => void;
/**
 * If x or x2 is not specified or 'center' 'left' 'right',
 * the width would be as long as possible.
 * If y or y2 is not specified or 'middle' 'top' 'bottom',
 * the height would be as long as possible.
 */
export declare function getAvailableSize(positionInfo: {
    left?: number | string;
    top?: number | string;
    right?: number | string;
    bottom?: number | string;
}, containerRect: {
    width: number;
    height: number;
}, margin?: number[] | number): {
    width: number;
    height: number;
};
/**
 * Parse position info.
 */
export declare function getLayoutRect(positionInfo: BoxLayoutOptionMixin & {
    aspect?: number;
}, containerRect: {
    width: number;
    height: number;
}, margin?: number | number[]): LayoutRect;
/**
 * Position a zr element in viewport
 *  Group position is specified by either
 *  {left, top}, {right, bottom}
 *  If all properties exists, right and bottom will be igonred.
 *
 * Logic:
 *     1. Scale (against origin point in parent coord)
 *     2. Rotate (against origin point in parent coord)
 *     3. Translate (with el.position by this method)
 * So this method only fixes the last step 'Translate', which does not affect
 * scaling and rotating.
 *
 * If be called repeatedly with the same input el, the same result will be gotten.
 *
 * Return true if the layout happened.
 *
 * @param el Should have `getBoundingRect` method.
 * @param positionInfo
 * @param positionInfo.left
 * @param positionInfo.top
 * @param positionInfo.right
 * @param positionInfo.bottom
 * @param positionInfo.width Only for opt.boundingModel: 'raw'
 * @param positionInfo.height Only for opt.boundingModel: 'raw'
 * @param containerRect
 * @param margin
 * @param opt
 * @param opt.hv Only horizontal or only vertical. Default to be [1, 1]
 * @param opt.boundingMode
 *        Specify how to calculate boundingRect when locating.
 *        'all': Position the boundingRect that is transformed and uioned
 *               both itself and its descendants.
 *               This mode simplies confine the elements in the bounding
 *               of their container (e.g., using 'right: 0').
 *        'raw': Position the boundingRect that is not transformed and only itself.
 *               This mode is useful when you want a element can overflow its
 *               container. (Consider a rotated circle needs to be located in a corner.)
 *               In this mode positionInfo.width/height can only be number.
 */
export declare function positionElement(el: Element, positionInfo: BoxLayoutOptionMixin, containerRect: {
    width: number;
    height: number;
}, margin?: number[] | number, opt?: {
    hv: [1 | 0 | boolean, 1 | 0 | boolean];
    boundingMode: 'all' | 'raw';
}, out?: {
    x?: number;
    y?: number;
}): boolean;
/**
 * @param option Contains some of the properties in HV_NAMES.
 * @param hvIdx 0: horizontal; 1: vertical.
 */
export declare function sizeCalculable(option: BoxLayoutOptionMixin, hvIdx: number): boolean;
export declare function fetchLayoutMode(ins: any): ComponentLayoutMode;
/**
 * Consider Case:
 * When default option has {left: 0, width: 100}, and we set {right: 0}
 * through setOption or media query, using normal zrUtil.merge will cause
 * {right: 0} does not take effect.
 *
 * @example
 * ComponentModel.extend({
 *     init: function () {
 *         ...
 *         let inputPositionParams = layout.getLayoutParams(option);
 *         this.mergeOption(inputPositionParams);
 *     },
 *     mergeOption: function (newOption) {
 *         newOption && zrUtil.merge(thisOption, newOption, true);
 *         layout.mergeLayoutParam(thisOption, newOption);
 *     }
 * });
 *
 * @param targetOption
 * @param newOption
 * @param opt
 */
export declare function mergeLayoutParam<T extends BoxLayoutOptionMixin>(targetOption: T, newOption: T, opt?: ComponentLayoutMode): void;
/**
 * Retrieve 'left', 'right', 'top', 'bottom', 'width', 'height' from object.
 */
export declare function getLayoutParams(source: BoxLayoutOptionMixin): BoxLayoutOptionMixin;
/**
 * Retrieve 'left', 'right', 'top', 'bottom', 'width', 'height' from object.
 * @param {Object} source
 * @return {Object} Result contains those props.
 */
export declare function copyLayoutParams(target: BoxLayoutOptionMixin, source: BoxLayoutOptionMixin): BoxLayoutOptionMixin;
export {};
