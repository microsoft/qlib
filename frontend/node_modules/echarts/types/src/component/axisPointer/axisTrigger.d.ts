import GlobalModel from '../../model/Global.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
import { Payload, CommonAxisPointerOption } from '../../util/types.js';
import { AxisPointerOption } from './AxisPointerModel.js';
declare type AxisValue = CommonAxisPointerOption['value'];
interface DataIndex {
    seriesIndex: number;
    dataIndex: number;
    dataIndexInside: number;
}
export interface DataByAxis {
    value: string | number;
    axisIndex: number;
    axisDim: string;
    axisType: string;
    axisId: string;
    seriesDataIndices: DataIndex[];
    valueLabelOpt: {
        precision: AxisPointerOption['label']['precision'];
        formatter: AxisPointerOption['label']['formatter'];
    };
}
export interface DataByCoordSys {
    coordSysId: string;
    coordSysIndex: number;
    coordSysType: string;
    coordSysMainType: string;
    dataByAxis: DataByAxis[];
}
interface AxisTriggerPayload extends Payload {
    currTrigger?: 'click' | 'mousemove' | 'leave';
    /**
     * x and y, which are mandatory, specify a point to trigger axisPointer and tooltip.
     */
    x?: number;
    /**
     * x and y, which are mandatory, specify a point to trigger axisPointer and tooltip.
     */
    y?: number;
    /**
     * finder, optional, restrict target axes.
     */
    seriesIndex?: number;
    dataIndex: number;
    axesInfo?: {
        axisDim?: string;
        axisIndex?: number;
        value?: AxisValue;
    }[];
    dispatchAction: ExtensionAPI['dispatchAction'];
}
/**
 * Basic logic: check all axis, if they do not demand show/highlight,
 * then hide/downplay them.
 *
 * @return content of event obj for echarts.connect.
 */
export default function axisTrigger(payload: AxisTriggerPayload, ecModel: GlobalModel, api: ExtensionAPI): AxisTriggerPayload;
export {};
