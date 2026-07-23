import Element from 'zrender/lib/Element.js';
import { DataModel, ECEventData, BlurScope, InnerFocus, SeriesDataType, ComponentMainType, ComponentItemTooltipOption } from './types.js';
export declare type SSRItemType = 'chart' | 'legend';
/**
 * ECData stored on graphic element
 */
export interface ECData {
    dataIndex?: number;
    dataModel?: DataModel;
    eventData?: ECEventData;
    seriesIndex?: number;
    dataType?: SeriesDataType;
    focus?: InnerFocus;
    blurScope?: BlurScope;
    ssrType?: SSRItemType;
    componentMainType?: ComponentMainType;
    componentIndex?: number;
    componentHighDownName?: string;
    tooltipConfig?: {
        name: string;
        option: ComponentItemTooltipOption<unknown>;
    };
}
export declare const getECData: (hostObj: Element<import("zrender/lib/Element").ElementProps>) => ECData;
export declare const setCommonECData: (seriesIndex: number, dataType: SeriesDataType, dataIdx: number, el: Element) => void;
