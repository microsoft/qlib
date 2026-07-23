import SeriesModel from '../../model/Series.js';
import { SeriesOption, SeriesEncodeOptionMixin, LineStyleOption, SeriesLabelOption, SeriesTooltipOption, OptionDataValue, StatesOptionMixin, DefaultStatesMixinEmphasis, ZRColor, CallbackDataParams } from '../../util/types.js';
import GlobalModel from '../../model/Global.js';
import SeriesData from '../../data/SeriesData.js';
import { ParallelActiveState, ParallelAxisOption } from '../../coord/parallel/AxisModel.js';
import Parallel from '../../coord/parallel/Parallel.js';
declare type ParallelSeriesDataValue = OptionDataValue[];
interface ParallelStatesMixin {
    emphasis?: DefaultStatesMixinEmphasis;
}
export interface ParallelStateOption<TCbParams = never> {
    lineStyle?: LineStyleOption<(TCbParams extends never ? never : (params: TCbParams) => ZRColor) | ZRColor>;
    label?: SeriesLabelOption;
}
export interface ParallelSeriesDataItemOption extends ParallelStateOption, StatesOptionMixin<ParallelStateOption, ParallelStatesMixin> {
    value?: ParallelSeriesDataValue;
}
export interface ParallelSeriesOption extends SeriesOption<ParallelStateOption<CallbackDataParams>, ParallelStatesMixin>, ParallelStateOption<CallbackDataParams>, SeriesEncodeOptionMixin {
    type?: 'parallel';
    coordinateSystem?: string;
    parallelIndex?: number;
    parallelId?: string;
    inactiveOpacity?: number;
    activeOpacity?: number;
    smooth?: boolean | number;
    realtime?: boolean;
    tooltip?: SeriesTooltipOption;
    parallelAxisDefault?: ParallelAxisOption;
    data?: (ParallelSeriesDataValue | ParallelSeriesDataItemOption)[];
}
declare class ParallelSeriesModel extends SeriesModel<ParallelSeriesOption> {
    static type: string;
    readonly type: string;
    static dependencies: string[];
    visualStyleAccessPath: string;
    visualDrawType: "stroke";
    coordinateSystem: Parallel;
    getInitialData(this: ParallelSeriesModel, option: ParallelSeriesOption, ecModel: GlobalModel): SeriesData;
    /**
     * User can get data raw indices on 'axisAreaSelected' event received.
     *
     * @return Raw indices
     */
    getRawIndicesByActiveState(activeState: ParallelActiveState): number[];
    static defaultOption: ParallelSeriesOption;
}
export default ParallelSeriesModel;
