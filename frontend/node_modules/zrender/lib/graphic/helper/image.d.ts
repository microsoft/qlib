import { ImageLike } from '../../core/types';
export declare function findExistImage(newImageOrSrc: string | ImageLike): ImageLike;
export declare function createOrUpdateImage<T>(newImageOrSrc: string | ImageLike, image: ImageLike, hostEl: {
    dirty: () => void;
}, onload?: (image: ImageLike, payload: T) => void, cbPayload?: T): ImageLike;
export declare function isImageReady(image: ImageLike): number;
