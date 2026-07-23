import { DataHost, DisplayState, CallbackDataParams, OptionDataValue, SeriesDataType, ComponentMainType, ComponentSubType, InterpolatableValue } from '../../util/types.js';
import GlobalModel from '../Global.js';
import { TooltipMarkupBlockFragment } from '../../component/tooltip/tooltipMarkup.js';
export interface DataFormatMixin extends DataHost {
    ecModel: GlobalModel;
    mainType: ComponentMainType;
    subType: ComponentSubType;
    componentIndex: number;
    id: string;
    name: string;
    animatedValue: OptionDataValue[];
}
export declare class DataFormatMixin {
    /**
     * Get params for formatter
     */
    getDataParams(dataIndex: number, dataType?: SeriesDataType): CallbackDataParams;
    /**
     * Format label
     * @param dataIndex
     * @param status 'normal' by default
     * @param dataType
     * @param labelDimIndex Only used in some chart that
     *        use formatter in different dimensions, like radar.
     * @param formatter Formatter given outside.
     * @return return null/undefined if no formatter
     */
    getFormattedLabel(dataIndex: number, status?: DisplayState, dataType?: SeriesDataType, labelDimIndex?: number, formatter?: string | ((params: object) => string), extendParams?: {
        interpolatedValue: InterpolatableValue;
    }): string;
    /**
     * Get raw value in option
     */
    getRawValue(idx: number, dataType?: SeriesDataType): unknown;
    /**
     * Should be implemented.
     * @param {number} dataIndex
     * @param {boolean} [multipleSeries=false]
     * @param {string} [dataType]
     */
    formatTooltip(dataIndex: number, multipleSeries?: boolean, dataType?: string): TooltipFormatResult;
}
declare type TooltipFormatResult = string | TooltipMarkupBlockFragment;
/**
 * For backward compat, normalize the return from `formatTooltip`.
 */
export declare function normalizeTooltipFormatResult(result: TooltipFormatResult): {
    frag: TooltipMarkupBlockFragment;
    text: string;
};
export {};
