import Geo from './Geo.js';
import { GeoOption, RegionOption } from './GeoModel.js';
import { MapSeriesOption } from '../../chart/map/MapSeries.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
import { CoordinateSystemCreator } from '../CoordinateSystem.js';
import { NameMap } from './geoTypes.js';
import type GlobalModel from '../../model/Global.js';
import type ComponentModel from '../../model/Component.js';
export declare type resizeGeoType = typeof resizeGeo;
/**
 * Resize method bound to the geo
 */
declare function resizeGeo(this: Geo, geoModel: ComponentModel<GeoOption | MapSeriesOption>, api: ExtensionAPI): void;
declare class GeoCreator implements CoordinateSystemCreator {
    dimensions: string[];
    create(ecModel: GlobalModel, api: ExtensionAPI): Geo[];
    /**
     * Fill given regions array
     */
    getFilledRegions(originRegionArr: RegionOption[], mapName: string, nameMap: NameMap, nameProperty: string): RegionOption[];
}
declare const geoCreator: GeoCreator;
export default geoCreator;
