import BoundingRect from 'zrender/lib/core/BoundingRect.js';
import { HashMap } from 'zrender/lib/core/util.js';
import { Group } from '../../util/graphic.js';
import { Region } from './Region.js';
export declare type GeoSVGSourceInput = string | Document | SVGElement;
export declare type GeoJSONSourceInput = string | GeoJSON | GeoJSONCompressed;
export interface NameMap {
    [regionName: string]: string;
}
export interface GeoSpecialAreas {
    [areaName: string]: {
        left: number;
        top: number;
        width?: number;
        height?: number;
    };
}
export interface GeoJSON extends GeoJSONFeatureCollection<GeoJSONGeometry> {
}
export interface GeoJSONCompressed extends GeoJSONFeatureCollection<GeoJSONGeometryCompressed> {
    UTF8Encoding?: boolean;
    UTF8Scale?: number;
}
interface GeoJSONFeatureCollection<G> {
    type: 'FeatureCollection';
    features: GeoJSONFeature<G>[];
}
interface GeoJSONFeature<G = GeoJSONGeometry> {
    type: 'Feature';
    id?: string | number;
    properties: {
        name?: string;
        cp?: number[];
        [key: string]: any;
    };
    geometry: G;
}
declare type GeoJSONGeometry = GeoJSONGeometryPoint | GeoJSONGeometryMultiPoint | GeoJSONGeometryLineString | GeoJSONGeometryMultiLineString | GeoJSONGeometryPolygon | GeoJSONGeometryMultiPolygon;
declare type GeoJSONGeometryCompressed = GeoJSONGeometryPolygonCompressed | GeoJSONGeometryMultiPolygonCompressed | GeoJSONGeometryLineStringCompressed | GeoJSONGeometryMultiLineStringCompressed;
interface GeoJSONGeometryPoint {
    type: 'Point';
    coordinates: number[];
}
interface GeoJSONGeometryMultiPoint {
    type: 'MultiPoint';
    coordinates: number[][];
}
interface GeoJSONGeometryLineString {
    type: 'LineString';
    coordinates: number[][];
}
interface GeoJSONGeometryLineStringCompressed {
    type: 'LineString';
    coordinates: string;
    encodeOffsets: number[];
}
interface GeoJSONGeometryMultiLineString {
    type: 'MultiLineString';
    coordinates: number[][][];
}
interface GeoJSONGeometryMultiLineStringCompressed {
    type: 'MultiLineString';
    coordinates: string[];
    encodeOffsets: number[][];
}
export interface GeoJSONGeometryPolygon {
    type: 'Polygon';
    coordinates: number[][][];
}
interface GeoJSONGeometryPolygonCompressed {
    type: 'Polygon';
    coordinates: string[];
    encodeOffsets: number[][];
}
export interface GeoJSONGeometryMultiPolygon {
    type: 'MultiPolygon';
    coordinates: number[][][][];
}
interface GeoJSONGeometryMultiPolygonCompressed {
    type: 'MultiPolygon';
    coordinates: string[][];
    encodeOffsets: number[][][];
}
export interface GeoResource {
    readonly type: 'geoJSON' | 'geoSVG';
    load(nameMap: NameMap, nameProperty: string): {
        boundingRect: BoundingRect;
        regions: Region[];
        regionsMap: HashMap<Region>;
    };
}
export interface GeoSVGGraphicRoot extends Group {
    isGeoSVGGraphicRoot: boolean;
}
/**
 * Geo stream interface compatitable with d3-geo
 * See the API detail in https://github.com/d3/d3-geo#streams
 */
export interface ProjectionStream {
    point(x: number, y: number): void;
    lineStart(): void;
    lineEnd(): void;
    polygonStart(): void;
    polygonEnd(): void;
    /**
     * Not supported yet.
     */
    sphere(): void;
}
export interface GeoProjection {
    project(point: number[]): number[];
    unproject(point: number[]): number[];
    /**
     * Projection stream compatitable to d3-geo projection stream.
     *
     * When rotate projection is used. It may have antimeridian artifacts.
     * So we need to introduce the fule projection stream to do antimeridian clipping.
     *
     * project will be ignored if projectStream is given.
     */
    stream?(outStream: ProjectionStream): ProjectionStream;
}
export {};
