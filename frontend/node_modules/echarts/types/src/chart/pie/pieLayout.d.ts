import GlobalModel from '../../model/Global.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
import PieSeriesModel from './PieSeries.js';
import { SectorShape } from 'zrender/lib/graphic/shape/Sector.js';
export declare function getBasicPieLayout(seriesModel: PieSeriesModel, api: ExtensionAPI): Pick<SectorShape, 'cx' | 'cy' | 'r' | 'r0'>;
export default function pieLayout(seriesType: 'pie', ecModel: GlobalModel, api: ExtensionAPI): void;
export declare const getSeriesLayoutData: (hostObj: PieSeriesModel) => {
    startAngle: number;
    endAngle: number;
    clockwise: boolean;
};
