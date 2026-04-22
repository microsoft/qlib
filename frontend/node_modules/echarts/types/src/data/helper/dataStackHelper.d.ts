import SeriesDimensionDefine from '../SeriesDimensionDefine.js';
import SeriesModel from '../../model/Series.js';
import SeriesData, { DataCalculationInfo } from '../SeriesData.js';
import type { SeriesOption, SeriesStackOptionMixin, DimensionName } from '../../util/types.js';
import { SeriesDataSchema } from './SeriesDataSchema.js';
import DataStore from '../DataStore.js';
declare type EnableDataStackDimensionsInput = {
    schema: SeriesDataSchema;
    store?: DataStore;
};
declare type EnableDataStackDimensionsInputLegacy = (SeriesDimensionDefine | string)[];
/**
 * Note that it is too complicated to support 3d stack by value
 * (have to create two-dimension inverted index), so in 3d case
 * we just support that stacked by index.
 *
 * @param seriesModel
 * @param dimensionsInput The same as the input of <module:echarts/data/SeriesData>.
 *        The input will be modified.
 * @param opt
 * @param opt.stackedCoordDimension Specify a coord dimension if needed.
 * @param opt.byIndex=false
 * @return calculationInfo
 * {
 *     stackedDimension: string
 *     stackedByDimension: string
 *     isStackedByIndex: boolean
 *     stackedOverDimension: string
 *     stackResultDimension: string
 * }
 */
export declare function enableDataStack(seriesModel: SeriesModel<SeriesOption & SeriesStackOptionMixin>, dimensionsInput: EnableDataStackDimensionsInput | EnableDataStackDimensionsInputLegacy, opt?: {
    stackedCoordDimension?: string;
    byIndex?: boolean;
}): Pick<DataCalculationInfo<unknown>, 'stackedDimension' | 'stackedByDimension' | 'isStackedByIndex' | 'stackedOverDimension' | 'stackResultDimension'>;
export declare function isDimensionStacked(data: SeriesData, stackedDim: string): boolean;
export declare function getStackedDimension(data: SeriesData, targetDim: string): DimensionName;
export {};
