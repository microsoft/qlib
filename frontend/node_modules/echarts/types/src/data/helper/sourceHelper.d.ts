import { Source } from '../Source.js';
import { DimensionName, OptionEncode, DimensionIndex, SeriesEncodableModel } from '../../util/types.js';
import { DatasetModel } from '../../component/dataset/install.js';
import SeriesModel from '../../model/Series.js';
import GlobalModel from '../../model/Global.js';
import { CoordDimensionDefinition } from './createDimensions.js';
export declare const BE_ORDINAL: {
    Must: number;
    Might: number;
    Not: number;
};
declare type BeOrdinalValue = (typeof BE_ORDINAL)[keyof typeof BE_ORDINAL];
declare type SeriesEncodeInternal = {
    [key in keyof OptionEncode]: DimensionIndex[];
};
/**
 * MUST be called before mergeOption of all series.
 */
export declare function resetSourceDefaulter(ecModel: GlobalModel): void;
/**
 * [The strategy of the arrengment of data dimensions for dataset]:
 * "value way": all axes are non-category axes. So series one by one take
 *     several (the number is coordSysDims.length) dimensions from dataset.
 *     The result of data arrengment of data dimensions like:
 *     | ser0_x | ser0_y | ser1_x | ser1_y | ser2_x | ser2_y |
 * "category way": at least one axis is category axis. So the the first data
 *     dimension is always mapped to the first category axis and shared by
 *     all of the series. The other data dimensions are taken by series like
 *     "value way" does.
 *     The result of data arrengment of data dimensions like:
 *     | ser_shared_x | ser0_y | ser1_y | ser2_y |
 *
 * @return encode Never be `null/undefined`.
 */
export declare function makeSeriesEncodeForAxisCoordSys(coordDimensions: (DimensionName | CoordDimensionDefinition)[], seriesModel: SeriesModel, source: Source): SeriesEncodeInternal;
/**
 * Work for data like [{name: ..., value: ...}, ...].
 *
 * @return encode Never be `null/undefined`.
 */
export declare function makeSeriesEncodeForNameBased(seriesModel: SeriesModel, source: Source, dimCount: number): SeriesEncodeInternal;
/**
 * @return If return null/undefined, indicate that should not use datasetModel.
 */
export declare function querySeriesUpstreamDatasetModel(seriesModel: SeriesEncodableModel): DatasetModel;
/**
 * @return Always return an array event empty.
 */
export declare function queryDatasetUpstreamDatasetModels(datasetModel: DatasetModel): DatasetModel[];
/**
 * The rule should not be complex, otherwise user might not
 * be able to known where the data is wrong.
 * The code is ugly, but how to make it neat?
 */
export declare function guessOrdinal(source: Source, dimIndex: DimensionIndex): BeOrdinalValue;
export {};
