import * as graphic from '../../util/graphic.js';
import { AxisBaseModel } from '../../coord/AxisBaseModel.js';
declare type AxisIndexKey = 'xAxisIndex' | 'yAxisIndex' | 'radiusAxisIndex' | 'angleAxisIndex' | 'singleAxisIndex';
declare type AxisEventData = {
    componentType: string;
    componentIndex: number;
    targetType: 'axisName' | 'axisLabel';
    name?: string;
    value?: string | number;
    dataIndex?: number;
    tickIndex?: number;
} & {
    [key in AxisIndexKey]?: number;
};
export interface AxisBuilderCfg {
    position?: number[];
    rotation?: number;
    /**
     * Used when nameLocation is 'middle' or 'center'.
     * 1 | -1
     */
    nameDirection?: number;
    tickDirection?: number;
    labelDirection?: number;
    /**
     * Useful when onZero.
     */
    labelOffset?: number;
    /**
     * default get from axisModel.
     */
    axisLabelShow?: boolean;
    /**
     * default get from axisModel.
     */
    axisName?: string;
    axisNameAvailableWidth?: number;
    /**
     * by degree, default get from axisModel.
     */
    labelRotate?: number;
    strokeContainThreshold?: number;
    nameTruncateMaxWidth?: number;
    silent?: boolean;
    handleAutoShown?(elementType: 'axisLine' | 'axisTick'): boolean;
}
/**
 * A final axis is translated and rotated from a "standard axis".
 * So opt.position and opt.rotation is required.
 *
 * A standard axis is and axis from [0, 0] to [0, axisExtent[1]],
 * for example: (0, 0) ------------> (0, 50)
 *
 * nameDirection or tickDirection or labelDirection is 1 means tick
 * or label is below the standard axis, whereas is -1 means above
 * the standard axis. labelOffset means offset between label and axis,
 * which is useful when 'onZero', where axisLabel is in the grid and
 * label in outside grid.
 *
 * Tips: like always,
 * positive rotation represents anticlockwise, and negative rotation
 * represents clockwise.
 * The direction of position coordinate is the same as the direction
 * of screen coordinate.
 *
 * Do not need to consider axis 'inverse', which is auto processed by
 * axis extent.
 */
declare class AxisBuilder {
    axisModel: AxisBaseModel;
    opt: AxisBuilderCfg;
    readonly group: graphic.Group;
    private _transformGroup;
    constructor(axisModel: AxisBaseModel, opt?: AxisBuilderCfg);
    hasBuilder(name: keyof typeof builders): boolean;
    add(name: keyof typeof builders): void;
    getGroup(): graphic.Group;
    static innerTextLayout(axisRotation: number, textRotation: number, direction: number): {
        rotation: number;
        textAlign: import("zrender/lib/core/types").TextAlign;
        textVerticalAlign: import("zrender/lib/core/types").TextVerticalAlign;
    };
    static makeAxisEventDataBase(axisModel: AxisBaseModel): AxisEventData;
    static isLabelSilent(axisModel: AxisBaseModel): boolean;
}
interface AxisElementsBuilder {
    (opt: AxisBuilderCfg, axisModel: AxisBaseModel, group: graphic.Group, transformGroup: graphic.Group): void;
}
declare const builders: Record<'axisLine' | 'axisTickLabel' | 'axisName', AxisElementsBuilder>;
export default AxisBuilder;
