import * as zrUtil from 'zrender/lib/core/util.js';
import { encodeHTML } from 'zrender/lib/core/dom.js';
import { TooltipRenderMode, ColorString, ZRColor, DimensionType } from './types.js';
import { Dictionary } from 'zrender/lib/core/types.js';
/**
 * Add a comma each three digit.
 */
export declare function addCommas(x: string | number): string;
export declare function toCamelCase(str: string, upperCaseFirst?: boolean): string;
export declare const normalizeCssArray: typeof zrUtil.normalizeCssArray;
export { encodeHTML };
/**
 * Make value user readable for tooltip and label.
 * "User readable":
 *     Try to not print programmer-specific text like NaN, Infinity, null, undefined.
 *     Avoid to display an empty string, which users can not recognize there is
 *     a value and it might look like a bug.
 */
export declare function makeValueReadable(value: unknown, valueType: DimensionType, useUTC: boolean): string;
export interface TplFormatterParam extends Dictionary<any> {
    $vars: string[];
}
/**
 * Template formatter
 * @param {Array.<Object>|Object} paramsList
 */
export declare function formatTpl(tpl: string, paramsList: TplFormatterParam | TplFormatterParam[], encode?: boolean): string;
/**
 * simple Template formatter
 */
export declare function formatTplSimple(tpl: string, param: Dictionary<any>, encode?: boolean): string;
interface RichTextTooltipMarker {
    renderMode: TooltipRenderMode;
    content: string;
    style: Dictionary<unknown>;
}
export declare type TooltipMarker = string | RichTextTooltipMarker;
export declare type TooltipMarkerType = 'item' | 'subItem';
interface GetTooltipMarkerOpt {
    color?: ColorString;
    extraCssText?: string;
    type?: TooltipMarkerType;
    renderMode?: TooltipRenderMode;
    markerId?: string;
}
export declare function getTooltipMarker(color: ColorString, extraCssText?: string): TooltipMarker;
export declare function getTooltipMarker(opt: GetTooltipMarkerOpt): TooltipMarker;
/**
 * @deprecated Use `time/format` instead.
 * ISO Date format
 * @param {string} tpl
 * @param {number} value
 * @param {boolean} [isUTC=false] Default in local time.
 *           see `module:echarts/scale/Time`
 *           and `module:echarts/util/number#parseDate`.
 * @inner
 */
export declare function formatTime(tpl: string, value: unknown, isUTC?: boolean): string;
/**
 * Capital first
 * @param {string} str
 * @return {string}
 */
export declare function capitalFirst(str: string): string;
/**
 * @return Never be null/undefined.
 */
export declare function convertToColorString(color: ZRColor, defaultColor?: ColorString): ColorString;
export { truncateText } from 'zrender/lib/graphic/helper/parseText.js';
/**
 * open new tab
 * @param link url
 * @param target blank or self
 */
export declare function windowOpen(link: string, target: string): void;
export { getTextRect } from '../legacy/getTextRect.js';
