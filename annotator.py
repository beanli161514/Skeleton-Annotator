import numpy as np
import napari
import os
from brightest_path_lib.algorithm import NBAStarSearch
from tifffile import imread, imwrite
from magicgui import widgets


class Annotator:
    def __init__(self):
        self.viewer = napari.Viewer(ndisplay=3,title='instance annotator')
        self.image_layer = self.viewer.add_image(np.zeros((64, 64, 64), dtype=np.uint16),name='image')
        self.start_layer = self.viewer.add_points(ndim=3,face_color='cyan',size=2,edge_color='black',shading='spherical',name='start')
        self.goal_layer = self.viewer.add_points(ndim=3,face_color='red',size=2,edge_color='black',shading='spherical',name='goal')
        self.path_layer = self.viewer.add_points(ndim=3,face_color='green',size=1,edge_color='black',shading='spherical',name='path')
        self.labeled_layer = self.viewer.add_points(ndim=3,size=0.8,edge_color='black',shading='spherical',face_colormap='turbo',name='saved labels')
        self.labeled_path = []
        self.add_callback()
        napari.run()


    def add_callback(self):
        self.viewer.bind_key('f', self.find_path)
        self.viewer.bind_key('r', self.delete_one_path)
        self.viewer.bind_key('d', self.delete_current_path)
        self.viewer.bind_key('s', self.save_current_path)
        self.image_layer.mouse_double_click_callbacks.append(self.on_double_click)

        self.button0 = widgets.PushButton(text="refresh")
        self.button0.clicked.connect(self.refresh)
        self.button1 = widgets.PushButton(text="save path (s)")
        self.button1.clicked.connect(self.save_current_path)
        self.button2 = widgets.PushButton(text="delete current path (d)")
        self.button2.clicked.connect(self.delete_current_path)
        self.button3 = widgets.PushButton(text="delete added path (r)")
        self.button3.clicked.connect(self.delete_one_path)
        self.button4 = widgets.PushButton(text="find path (f)")
        self.button4.clicked.connect(self.find_path)
        self.button5 = widgets.PushButton(text="save mask")
        self.button5.clicked.connect(self.save_mask)

        self.image_path = widgets.FileEdit(label="image_path")
        self.container = widgets.Container(widgets=[self.image_path, self.button0,self.button5,self.button1,self.button2,self.button3,self.button4])
        self.viewer.window.add_dock_widget(self.container, area='right')


    def save_current_path(self,viewer):
        if len(self.path_layer.data)==0:
            return
        self.labeled_path.append(self.path_layer.data.tolist())
        colors = []
        points = []
        scolors = [i/len(self.labeled_path) for i in list(range(len(self.labeled_path)))]
        for i, seg in enumerate(self.labeled_path):
            seg_color = scolors[i]
            for point in seg:
                points.append(point)
                colors.append(seg_color)
        
        # to avoid 1/0 when scaling values
        if colors[-1] == 0:
            colors[-1] = 1

        properties = {
            'colors': np.array(colors,dtype=np.float32)
        }

        self.labeled_layer.data = np.array(points)
        self.labeled_layer.properties = properties
        self.labeled_layer.face_color = 'colors'
        self.labeled_layer.face_colormap = 'turbo'
        self.labeled_layer.selected_data = []
        self.path_layer.data = np.array([])
        self.labeled_layer.refresh()
        self.start_layer.data = []
        self.goal_layer.data = []
        

    def delete_one_path(self,viewer):
        if len(self.labeled_path)<=1:
            self.labeled_path = []
        else:
            self.labeled_path = self.labeled_path[:-1]
        colors = []
        points = []
        scolors = [i/len(self.labeled_path) for i in list(range(len(self.labeled_path)))]
        for i, seg in enumerate(self.labeled_path):
            seg_color = scolors[i]
            for point in seg:
                points.append(point)
                colors.append(seg_color)
        
        if len(colors)!=0:
            if colors[-1] == 0:
                colors[-1] = 1

        properties = {
            'colors': np.array(colors,dtype=np.float32)
        }

        self.labeled_layer.data = np.array(points)
        self.labeled_layer.properties = properties
        self.labeled_layer.face_color = 'colors'
        self.labeled_layer.face_colormap = 'turbo'
        self.labeled_layer.selected_data = []
        self.path_layer.data = np.array([])
        self.labeled_layer.refresh()
        self.start_layer.data = []
        self.goal_layer.data = []


    def refresh(self):
        img = imread(self.image_path.value)
        self.labeled_path = []
        self.path_layer.data = []
        self.labeled_layer.data = []
        self.image_layer.data = img
        self.viewer.reset_view()
        self.image_layer.reset_contrast_limits()
        self.viewer.layers.selection.active = self.image_layer


    def find_path(self,viewer):
        sa = NBAStarSearch(self.image_layer.data, start_point=self.start_layer.data[0], goal_point=self.goal_layer.data[0])
        path = sa.search()
        if len(self.path_layer.data)!=0:
            total_path = np.concatenate((self.path_layer.data, np.array(path)), axis=0)
        else:
            total_path = path

        self.path_layer.data = total_path
        self.path_layer.selected_data = np.array([])
        self.path_layer.refresh()


    def delete_current_path(self,viewer):
        self.path_layer.data = []
        self.path_layer.selected_data = np.array([])
        self.path_layer.refresh()


    def step_forward(self,viewer):
        self.start_layer.data = self.goal_layer.data
        self.goal_layer.data = []


    def on_double_click(self,layer,event):
        #based on ray casting
        position, direction = self.map_click(event) 
        near_point, far_point = layer.get_ray_intersections(
            position,
            direction,
            event.dims_displayed
        )
        sample_ray = far_point - near_point
        length_sample_vector = np.linalg.norm(sample_ray)
        increment_vector = sample_ray / (2 * length_sample_vector)
        n_iterations = int(2 * length_sample_vector)
        bbox = np.array([
            [0, layer.data.shape[0]-1],
            [0, layer.data.shape[1]-1],
            [0, layer.data.shape[2]-1]
        ])
        sample_points = []
        values = []
        for i in range(n_iterations):
            sample_point = np.asarray(near_point + i * increment_vector, dtype=int)
            sample_point = self.clamp_point_to_bbox(sample_point, bbox)
            value = layer.data[sample_point[0], sample_point[1], sample_point[2]]
            sample_points.append(sample_point)
            values.append(value)
        max_point_index = values.index(max(values))
        max_point = sample_points[max_point_index]
        print('Put point at: ', max_point)
        if(event.button==2):
            self.start_layer.data = max_point
        if(event.button==1):
            self.goal_layer.data = max_point


    def clamp_point_to_bbox(self,point: np.ndarray, bbox: np.ndarray):
        clamped_point = np.clip(point, bbox[:, 0], bbox[:, 1])
        return clamped_point


    def save_mask(self,viewer):
        image_path = self.image_path.value
        directory, image_name = os.path.split(image_path)
        mask_name = image_name.replace('img_','mask_')


        image = self.image_layer.data
        path = sum(self.labeled_path,[])
        coordinates = np.array(path)
        mask = np.zeros(image.shape, dtype=np.uint8)
        if len(coordinates)>0:
            mask[coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]] = 1

        mask_path = os.path.join(directory, mask_name)

        imwrite(mask_path,mask,dtype=np.uint8)

        print(mask_path+' saved')


    def map_click(self,event):
        x, y = event.pos
        w, h = self.viewer.window.qt_viewer.canvas.size
        transform = self.viewer.window.qt_viewer.view.camera._scene_transform

        p0 = transform.imap([x,y,0,1]) # map click pos to scene coordinates
        p1 = [w/2,h/2,-1e10,1] # canvas center at infinite far z- (eye position in canvas coordinates)
        p1 = transform.imap(p1) # map eye pos to scene coordinates
        p0 = p0[0:3]/p0[3] # homogeneous coordinate to cartesian
        p1 = p1[0:3]/p1[3] # homogeneous coordinate to cartesian

        # calculate direction of the ray
        d = p1 - p0
        d = d[0:3]
        d = d / np.linalg.norm(d)

        p0 = list(p0[::-1]) # xyz to zyx
        d = list(d[::-1]) # xyz to zyx
        return p0, d


def main():
    anno = Annotator()


if __name__ == '__main__':
    main()
