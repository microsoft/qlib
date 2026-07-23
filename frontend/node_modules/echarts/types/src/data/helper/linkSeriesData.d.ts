import SeriesData from '../SeriesData.js';
import { SeriesDataType } from '../../util/types.js';
declare type Datas = {
    [key in SeriesDataType]?: SeriesData;
};
declare type StructReferDataAttr = 'data' | 'edgeData';
declare type StructAttr = 'tree' | 'graph';
declare type LinkSeriesDataOpt = {
    mainData: SeriesData;
    struct: {
        update: () => void;
    } & {
        [key in StructReferDataAttr]?: SeriesData;
    };
    structAttr: StructAttr;
    datas?: Datas;
    datasAttr?: {
        [key in SeriesDataType]?: StructReferDataAttr;
    };
};
declare function linkSeriesData(opt: LinkSeriesDataOpt): void;
export default linkSeriesData;
