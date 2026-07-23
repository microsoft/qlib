import { PrepareSeriesDataSchemaParams } from '../../data/helper/createDimensions.js';
import SeriesData from '../../data/SeriesData.js';
import SeriesModel from '../../model/Series.js';
/**
 * [Usage]:
 * (1)
 * createListSimply(seriesModel, ['value']);
 * (2)
 * createListSimply(seriesModel, {
 *     coordDimensions: ['value'],
 *     dimensionsCount: 5
 * });
 */
export default function createSeriesDataSimply(seriesModel: SeriesModel, opt: PrepareSeriesDataSchemaParams | PrepareSeriesDataSchemaParams['coordDimensions'], nameList?: string[]): SeriesData;
