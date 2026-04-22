import View from '../View.js';
import { Region } from './Region.js';
import { GeoProjection, GeoResource, NameMap } from './geoTypes.js';
import GlobalModel from '../../model/Global.js';
import { ParsedModelFinder } from '../../util/model.js';
import GeoModel from './GeoModel.js';
import { resizeGeoType } from './geoCreator.js';
export declare const geo2DDimensions: string[];
declare class Geo extends View {
    dimensions: string[];
    type: string;
    readonly map: string;
    readonly resourceType: GeoResource['type'];
    private _nameCoordMap;
    private _regionsMap;
    private _invertLongitute;
    readonly regions: Region[];
    readonly aspectScale: number;
    projection: GeoProjection;
    model: GeoModel;
    resize: resizeGeoType;
    constructor(name: string, map: string, opt: {
        projection?: GeoProjection;
        nameMap?: NameMap;
        nameProperty?: string;
        aspectScale?: number;
    });
    protected _transformTo(x: number, y: number, width: number, height: number): void;
    getRegion(name: string): Region;
    getRegionByCoord(coord: number[]): Region;
    /**
     * Add geoCoord for indexing by name
     */
    addGeoCoord(name: string, geoCoord: number[]): void;
    /**
     * Get geoCoord by name
     */
    getGeoCoord(name: string): number[];
    dataToPoint(data: number[] | string, noRoam?: boolean, out?: number[]): number[];
    pointToData(point: number[]): number[];
    /**
     * Point to projected data. Same with pointToData when projection is used.
     */
    pointToProjected(point: number[]): number[];
    projectedToPoint(projected: number[], noRoam?: boolean, out?: number[]): number[];
    convertToPixel(ecModel: GlobalModel, finder: ParsedModelFinder, value: number[]): number[];
    convertFromPixel(ecModel: GlobalModel, finder: ParsedModelFinder, pixel: number[]): number[];
}
export default Geo;
