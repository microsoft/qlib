import ComponentModel from '../../model/Component.js';
import Parallel from './Parallel.js';
import { DimensionName, ComponentOption, BoxLayoutOptionMixin } from '../../util/types.js';
import ParallelAxisModel, { ParallelAxisOption } from './AxisModel.js';
import GlobalModel from '../../model/Global.js';
import SeriesModel from '../../model/Series.js';
export declare type ParallelLayoutDirection = 'horizontal' | 'vertical';
export interface ParallelCoordinateSystemOption extends ComponentOption, BoxLayoutOptionMixin {
    mainType?: 'parallel';
    layout?: ParallelLayoutDirection;
    axisExpandable?: boolean;
    axisExpandCenter?: number;
    axisExpandCount?: number;
    axisExpandWidth?: number;
    axisExpandTriggerOn?: 'click' | 'mousemove';
    axisExpandRate?: number;
    axisExpandDebounce?: number;
    axisExpandSlideTriggerArea?: [number, number, number];
    axisExpandWindow?: number[];
    parallelAxisDefault?: ParallelAxisOption;
}
declare class ParallelModel extends ComponentModel<ParallelCoordinateSystemOption> {
    static type: string;
    readonly type: string;
    static dependencies: string[];
    coordinateSystem: Parallel;
    /**
     * Each item like: 'dim0', 'dim1', 'dim2', ...
     */
    dimensions: DimensionName[];
    /**
     * Corresponding to dimensions.
     */
    parallelAxisIndex: number[];
    static layoutMode: "box";
    static defaultOption: ParallelCoordinateSystemOption;
    init(): void;
    mergeOption(newOption: ParallelCoordinateSystemOption): void;
    /**
     * Whether series or axis is in this coordinate system.
     */
    contains(model: SeriesModel | ParallelAxisModel, ecModel: GlobalModel): boolean;
    setAxisExpand(opt: {
        axisExpandable?: boolean;
        axisExpandCenter?: number;
        axisExpandCount?: number;
        axisExpandWidth?: number;
        axisExpandWindow?: number[];
    }): void;
    private _initDimensions;
}
export default ParallelModel;
