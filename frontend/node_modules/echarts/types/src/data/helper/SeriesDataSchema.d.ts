import { HashMap } from 'zrender/lib/core/util.js';
import { DimensionDefinition, DimensionDefinitionLoose, DimensionIndex, DimensionName } from '../../util/types.js';
import { DataStoreDimensionDefine } from '../DataStore.js';
import SeriesDimensionDefine from '../SeriesDimensionDefine.js';
import { Source } from '../Source.js';
/**
 * Represents the dimension requirement of a series.
 *
 * NOTICE:
 * When there are too many dimensions in dataset and many series, only the used dimensions
 * (i.e., used by coord sys and declared in `series.encode`) are add to `dimensionDefineList`.
 * But users may query data by other unused dimension names.
 * In this case, users can only query data if and only if they have defined dimension names
 * via ec option, so we provide `getDimensionIndexFromSource`, which only query them from
 * `source` dimensions.
 */
export declare class SeriesDataSchema {
    /**
     * When there are too many dimensions, `dimensionDefineList` might only contain
     * used dimensions.
     *
     * CAUTION:
     * Should have been sorted by `storeDimIndex` asc.
     *
     * PENDING:
     * The item can still be modified outsite.
     * But MUST NOT add/remove item of this array.
     */
    readonly dimensions: SeriesDimensionDefine[];
    readonly source: Source;
    private _fullDimCount;
    private _dimNameMap;
    private _dimOmitted;
    constructor(opt: {
        source: Source;
        dimensions: SeriesDimensionDefine[];
        fullDimensionCount: number;
        dimensionOmitted: boolean;
    });
    isDimensionOmitted(): boolean;
    private _updateDimOmitted;
    /**
     * @caution Can only be used when `dimensionOmitted: true`.
     *
     * Get index by user defined dimension name (i.e., not internal generate name).
     * That is, get index from `dimensionsDefine`.
     * If no `dimensionsDefine`, or no name get, return -1.
     */
    getSourceDimensionIndex(dimName: DimensionName): DimensionIndex;
    /**
     * @caution Can only be used when `dimensionOmitted: true`.
     *
     * Notice: may return `null`/`undefined` if user not specify dimension names.
     */
    getSourceDimension(dimIndex: DimensionIndex): DimensionDefinition;
    makeStoreSchema(): {
        dimensions: DataStoreDimensionDefine[];
        hash: string;
    };
    makeOutputDimensionNames(): DimensionName[];
    appendCalculationDimension(dimDef: SeriesDimensionDefine): void;
}
export declare function isSeriesDataSchema(schema: any): schema is SeriesDataSchema;
export declare function createDimNameMap(dimsDef: DimensionDefinitionLoose[]): HashMap<DimensionIndex, DimensionName>;
export declare function ensureSourceDimNameMap(source: Source): HashMap<DimensionIndex, DimensionName>;
export declare function shouldOmitUnusedDimensions(dimCount: number): boolean;
