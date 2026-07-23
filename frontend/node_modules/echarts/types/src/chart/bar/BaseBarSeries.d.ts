import SeriesModel from '../../model/Series.js';
import { SeriesOption, SeriesOnCartesianOptionMixin, SeriesOnPolarOptionMixin, ScaleDataValue, DefaultStatesMixin, StatesMixinBase } from '../../util/types.js';
import GlobalModel from '../../model/Global.js';
import SeriesData from '../../data/SeriesData.js';
import { dimPermutations } from '../../component/marker/MarkAreaView.js';
export interface BaseBarSeriesOption<StateOption, ExtraStateOption extends StatesMixinBase = DefaultStatesMixin> extends SeriesOption<StateOption, ExtraStateOption>, SeriesOnCartesianOptionMixin, SeriesOnPolarOptionMixin {
    /**
     * Min height of bar
     */
    barMinHeight?: number;
    /**
     * Min angle of bar. Available on polar coordinate system.
     */
    barMinAngle?: number;
    /**
     * Max width of bar. Defaults to 1 on cartesian coordinate system. Otherwise it's null.
     */
    barMaxWidth?: number | string;
    barMinWidth?: number | string;
    /**
     * Bar width. Will be calculated automatically.
     * Can be pixel width or percent string.
     */
    barWidth?: number | string;
    /**
     * Gap between each bar inside category. Default to be 30%. Can be an aboslute pixel value
     */
    barGap?: string | number;
    /**
     * Gap between each category. Default to be 20%. can be an absolute pixel value.
     */
    barCategoryGap?: string | number;
    large?: boolean;
    largeThreshold?: number;
}
declare class BaseBarSeriesModel<Opts extends BaseBarSeriesOption<unknown> = BaseBarSeriesOption<unknown>> extends SeriesModel<Opts> {
    static type: string;
    type: string;
    getInitialData(option: Opts, ecModel: GlobalModel): SeriesData;
    getMarkerPosition(value: ScaleDataValue[], dims?: typeof dimPermutations[number], startingAtTick?: boolean): number[];
    static defaultOption: BaseBarSeriesOption<unknown, unknown>;
}
export default BaseBarSeriesModel;
