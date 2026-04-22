import GlobalModel from '../../model/Global.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
import { ThemeRiverSeriesOption } from './ThemeRiverSeries.js';
import { RectLike } from 'zrender/lib/core/BoundingRect.js';
export interface ThemeRiverLayoutInfo {
    rect: RectLike;
    boundaryGap: ThemeRiverSeriesOption['boundaryGap'];
}
export default function themeRiverLayout(ecModel: GlobalModel, api: ExtensionAPI): void;
