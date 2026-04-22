import SeriesData from '../../data/SeriesData.js';
import { OptionSourceData, EncodeDefaulter } from '../../util/types.js';
import SeriesModel from '../../model/Series.js';
/**
 * Caution: there are side effects to `sourceManager` in this method.
 * Should better only be called in `Series['getInitialData']`.
 */
declare function createSeriesData(sourceRaw: OptionSourceData | null | undefined, seriesModel: SeriesModel, opt?: {
    generateCoord?: string;
    useEncodeDefaulter?: boolean | EncodeDefaulter;
    createInvertedIndices?: boolean;
}): SeriesData;
export default createSeriesData;
