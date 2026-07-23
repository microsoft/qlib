import SeriesData from '../SeriesData.js';
import { DimensionName, DimensionIndex } from '../../util/types.js';
import { DataStoreDimensionType } from '../DataStore.js';
import { SeriesDataSchema } from './SeriesDataSchema.js';
export declare type DimensionSummaryEncode = {
    defaultedLabel: DimensionName[];
    defaultedTooltip: DimensionName[];
    [coordOrVisualDimName: string]: DimensionName[];
};
export declare type DimensionSummary = {
    encode: DimensionSummaryEncode;
    userOutput: DimensionUserOuput;
    dataDimsOnCoord: DimensionName[];
    dataDimIndicesOnCoord: DimensionIndex[];
    encodeFirstDimNotExtra: {
        [coordDim: string]: DimensionName;
    };
};
export declare type DimensionUserOuputEncode = {
    [coordOrVisualDimName: string]: DimensionIndex[];
};
declare class DimensionUserOuput {
    private _encode;
    private _cachedDimNames;
    private _schema?;
    constructor(encode: DimensionUserOuputEncode, dimRequest?: SeriesDataSchema);
    get(): {
        fullDimensions: DimensionName[];
        encode: DimensionUserOuputEncode;
    };
    /**
     * Get all data store dimension names.
     * Theoretically a series data store is defined both by series and used dataset (if any).
     * If some dimensions are omitted for performance reason in `this.dimensions`,
     * the dimension name may not be auto-generated if user does not specify a dimension name.
     * In this case, the dimension name is `null`/`undefined`.
     */
    private _getFullDimensionNames;
}
export declare function summarizeDimensions(data: SeriesData, schema?: SeriesDataSchema): DimensionSummary;
export declare function getDimensionTypeByAxis(axisType: string): DataStoreDimensionType;
export {};
