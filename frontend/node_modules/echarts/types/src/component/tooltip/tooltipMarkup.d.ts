import { Dictionary, TooltipRenderMode, ColorString, TooltipOrderMode, DimensionType, CommonTooltipOption } from '../../util/types.js';
import { TooltipMarkerType } from '../../util/format.js';
import SeriesModel from '../../model/Series.js';
import Model from '../../model/Model.js';
import { TooltipOption } from './TooltipModel.js';
/**
 * This is an abstract layer to insulate the upper usage of tooltip content
 * from the different backends according to different `renderMode` ('html' or 'richText').
 * With the help of the abstract layer, it does not need to consider how to create and
 * assemble html or richText snippets when making tooltip content.
 *
 * @usage
 *
 * ```ts
 * class XxxSeriesModel {
 *     formatTooltip(
 *         dataIndex: number,
 *         multipleSeries: boolean,
 *         dataType: string
 *     ) {
 *         ...
 *         return createTooltipMarkup('section', {
 *             header: header,
 *             blocks: [
 *                 createTooltipMarkup('nameValue', {
 *                     name: name,
 *                     value: value,
 *                     noValue: value == null
 *                 })
 *             ]
 *         });
 *     }
 * }
 * ```
 */
export declare type TooltipMarkupBlockFragment = TooltipMarkupSection | TooltipMarkupNameValueBlock;
interface TooltipMarkupBlock {
    sortParam?: unknown;
}
export interface TooltipMarkupSection extends TooltipMarkupBlock {
    type: 'section';
    header?: unknown;
    noHeader?: boolean;
    blocks?: TooltipMarkupBlockFragment[];
    sortBlocks?: boolean;
    valueFormatter?: CommonTooltipOption<unknown>['valueFormatter'];
}
export interface TooltipMarkupNameValueBlock extends TooltipMarkupBlock {
    type: 'nameValue';
    markerType?: TooltipMarkerType;
    markerColor?: ColorString;
    name?: string;
    value?: unknown | unknown[];
    valueType?: DimensionType | DimensionType[];
    noName?: boolean;
    noValue?: boolean;
    dataIndex?: number;
    valueFormatter?: CommonTooltipOption<unknown>['valueFormatter'];
}
/**
 * Create tooltip markup by this function, we can get TS type check.
 */
export declare function createTooltipMarkup(type: 'section', option: Omit<TooltipMarkupSection, 'type'>): TooltipMarkupSection;
export declare function createTooltipMarkup(type: 'nameValue', option: Omit<TooltipMarkupNameValueBlock, 'type'>): TooltipMarkupNameValueBlock;
declare type MarkupText = string;
/**
 * @return markupText. null/undefined means no content.
 */
export declare function buildTooltipMarkup(fragment: TooltipMarkupBlockFragment, markupStyleCreator: TooltipMarkupStyleCreator, renderMode: TooltipRenderMode, orderMode: TooltipOrderMode, useUTC: boolean, toolTipTextStyle: TooltipOption['textStyle']): MarkupText;
export declare function retrieveVisualColorForTooltipMarker(series: SeriesModel, dataIndex: number): ColorString;
export declare function getPaddingFromTooltipModel(model: Model<TooltipOption>, renderMode: TooltipRenderMode): number | number[];
/**
 * The major feature is generate styles for `renderMode: 'richText'`.
 * But it also serves `renderMode: 'html'` to provide
 * "renderMode-independent" API.
 */
export declare class TooltipMarkupStyleCreator {
    readonly richTextStyles: Dictionary<Dictionary<unknown>>;
    private _nextStyleNameId;
    private _generateStyleName;
    makeTooltipMarker(markerType: TooltipMarkerType, colorStr: ColorString, renderMode: TooltipRenderMode): string;
    /**
     * @usage
     * ```ts
     * const styledText = markupStyleCreator.wrapRichTextStyle([
     *     // The styles will be auto merged.
     *     {
     *         fontSize: 12,
     *         color: 'blue'
     *     },
     *     {
     *         padding: 20
     *     }
     * ]);
     * ```
     */
    wrapRichTextStyle(text: string, styles: Dictionary<unknown> | Dictionary<unknown>[]): string;
}
export {};
