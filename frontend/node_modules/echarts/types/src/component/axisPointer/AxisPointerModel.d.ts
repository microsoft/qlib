import ComponentModel from '../../model/Component.js';
import { ComponentOption, ScaleDataValue, CommonAxisPointerOption } from '../../util/types.js';
interface MapperParamAxisInfo {
    axisIndex: number;
    axisName: string;
    axisId: string;
    axisDim: string;
}
interface AxisPointerLink {
    xAxisIndex?: number[] | 'all';
    yAxisIndex?: number[] | 'all';
    xAxisId?: string[];
    yAxisId?: string[];
    xAxisName?: string[] | string;
    yAxisName?: string[] | string;
    radiusAxisIndex?: number[] | 'all';
    angleAxisIndex?: number[] | 'all';
    radiusAxisId?: string[];
    angleAxisId?: string[];
    radiusAxisName?: string[] | string;
    angleAxisName?: string[] | string;
    singleAxisIndex?: number[] | 'all';
    singleAxisId?: string[];
    singleAxisName?: string[] | string;
    mapper?(sourceVal: ScaleDataValue, sourceAxisInfo: MapperParamAxisInfo, targetAxisInfo: MapperParamAxisInfo): CommonAxisPointerOption['value'];
}
export interface AxisPointerOption extends ComponentOption, Omit<CommonAxisPointerOption, 'type'> {
    mainType?: 'axisPointer';
    type?: 'line' | 'shadow' | 'cross' | 'none';
    link?: AxisPointerLink[];
}
declare class AxisPointerModel extends ComponentModel<AxisPointerOption> {
    static type: "axisPointer";
    type: "axisPointer";
    coordSysAxesInfo: unknown;
    static defaultOption: AxisPointerOption;
}
export default AxisPointerModel;
