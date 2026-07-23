import { Dictionary } from 'zrender/lib/core/types.js';
import Displayable from 'zrender/lib/graphic/Displayable.js';
import Element, { ElementEvent } from 'zrender/lib/Element.js';
import Model from '../model/Model.js';
import { SeriesDataType, DisplayState, ECElement, BlurScope, InnerFocus, Payload, HighlightPayload, DownplayPayload, ComponentMainType } from './types.js';
import SeriesModel from '../model/Series.js';
import Path from 'zrender/lib/graphic/Path.js';
import GlobalModel from '../model/Global.js';
import ExtensionAPI from '../core/ExtensionAPI.js';
import ComponentModel from '../model/Component.js';
export declare const HOVER_STATE_NORMAL: 0;
export declare const HOVER_STATE_BLUR: 1;
export declare const HOVER_STATE_EMPHASIS: 2;
export declare const SPECIAL_STATES: readonly ["emphasis", "blur", "select"];
export declare const DISPLAY_STATES: readonly ["normal", "emphasis", "blur", "select"];
export declare const Z2_EMPHASIS_LIFT = 10;
export declare const Z2_SELECT_LIFT = 9;
export declare const HIGHLIGHT_ACTION_TYPE = "highlight";
export declare const DOWNPLAY_ACTION_TYPE = "downplay";
export declare const SELECT_ACTION_TYPE = "select";
export declare const UNSELECT_ACTION_TYPE = "unselect";
export declare const TOGGLE_SELECT_ACTION_TYPE = "toggleSelect";
export declare function setStatesFlag(el: ECElement, stateName: DisplayState): void;
/**
 * If we reuse elements when rerender.
 * DON'T forget to clearStates before we update the style and shape.
 * Or we may update on the wrong state instead of normal state.
 */
export declare function clearStates(el: Element): void;
/**
 * Set hover style (namely "emphasis style") of element.
 * @param el Should not be `zrender/graphic/Group`.
 * @param focus 'self' | 'selfInSeries' | 'series'
 */
export declare function setDefaultStateProxy(el: Displayable): void;
export declare function enterEmphasisWhenMouseOver(el: Element, e: ElementEvent): void;
export declare function leaveEmphasisWhenMouseOut(el: Element, e: ElementEvent): void;
export declare function enterEmphasis(el: Element, highlightDigit?: number): void;
export declare function leaveEmphasis(el: Element, highlightDigit?: number): void;
export declare function enterBlur(el: Element): void;
export declare function leaveBlur(el: Element): void;
export declare function enterSelect(el: Element): void;
export declare function leaveSelect(el: Element): void;
export declare function allLeaveBlur(api: ExtensionAPI): void;
export declare function blurSeries(targetSeriesIndex: number, focus: InnerFocus, blurScope: BlurScope, api: ExtensionAPI): void;
export declare function blurComponent(componentMainType: ComponentMainType, componentIndex: number, api: ExtensionAPI): void;
export declare function blurSeriesFromHighlightPayload(seriesModel: SeriesModel, payload: HighlightPayload, api: ExtensionAPI): void;
export declare function findComponentHighDownDispatchers(componentMainType: ComponentMainType, componentIndex: number, name: string, api: ExtensionAPI): {
    focusSelf: boolean;
    dispatchers: Element[];
};
export declare function handleGlobalMouseOverForHighDown(dispatcher: Element, e: ElementEvent, api: ExtensionAPI): void;
export declare function handleGlobalMouseOutForHighDown(dispatcher: Element, e: ElementEvent, api: ExtensionAPI): void;
export declare function toggleSelectionFromPayload(seriesModel: SeriesModel, payload: Payload, api: ExtensionAPI): void;
export declare function updateSeriesElementSelection(seriesModel: SeriesModel): void;
export declare function getAllSelectedIndices(ecModel: GlobalModel): {
    seriesIndex: number;
    dataType?: SeriesDataType;
    dataIndex: number[];
}[];
/**
 * Enable the function that mouseover will trigger the emphasis state.
 *
 * NOTE:
 * This function should be used on the element with dataIndex, seriesIndex.
 *
 */
export declare function enableHoverEmphasis(el: Element, focus?: InnerFocus, blurScope?: BlurScope): void;
export declare function disableHoverEmphasis(el: Element): void;
export declare function toggleHoverEmphasis(el: Element, focus: InnerFocus, blurScope: BlurScope, isDisabled: boolean): void;
export declare function enableHoverFocus(el: Element, focus: InnerFocus, blurScope: BlurScope): void;
/**
 * Set emphasis/blur/selected states of element.
 */
export declare function setStatesStylesFromModel(el: Displayable, itemModel: Model<Partial<Record<'emphasis' | 'blur' | 'select', any>>>, styleType?: string, // default itemStyle
getter?: (model: Model) => Dictionary<any>): void;
/**
 *
 * Set element as highlight / downplay dispatcher.
 * It will be checked when element received mouseover event or from highlight action.
 * It's in change of all highlight/downplay behavior of it's children.
 *
 * @param el
 * @param el.highDownSilentOnTouch
 *        In touch device, mouseover event will be trigger on touchstart event
 *        (see module:zrender/dom/HandlerProxy). By this mechanism, we can
 *        conveniently use hoverStyle when tap on touch screen without additional
 *        code for compatibility.
 *        But if the chart/component has select feature, which usually also use
 *        hoverStyle, there might be conflict between 'select-highlight' and
 *        'hover-highlight' especially when roam is enabled (see geo for example).
 *        In this case, `highDownSilentOnTouch` should be used to disable
 *        hover-highlight on touch device.
 * @param asDispatcher If `false`, do not set as "highDownDispatcher".
 */
export declare function setAsHighDownDispatcher(el: Element, asDispatcher: boolean): void;
export declare function isHighDownDispatcher(el: Element): boolean;
/**
 * Enable component highlight/downplay features:
 * + hover link (within the same name)
 * + focus blur in component
 */
export declare function enableComponentHighDownFeatures(el: Element, componentModel: ComponentModel, componentHighDownName: string): void;
/**
 * Support highlight/downplay record on each elements.
 * For the case: hover highlight/downplay (legend, visualMap, ...) and
 * user triggered highlight/downplay should not conflict.
 * Only all of the highlightDigit cleared, return to normal.
 * @param {string} highlightKey
 * @return {number} highlightDigit
 */
export declare function getHighlightDigit(highlightKey: number): number;
export declare function isSelectChangePayload(payload: Payload): boolean;
export declare function isHighDownPayload(payload: Payload): payload is HighlightPayload | DownplayPayload;
export declare function savePathStates(el: Path): void;
