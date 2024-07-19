import xml.etree.ElementTree as ET

def parse_tracklet_labels(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    labels = []

    for item in root.findall('.//item'):
        object_type = item.find('objectType').text
        for pose in item.find('poses'):
            tx = float(pose.find('tx').text)
            ty = float(pose.find('ty').text)
            tz = float(pose.find('tz').text)
            labels.append({
                'type': object_type,
                'tx': tx,
                'ty': ty,
                'tz': tz
            })

    return labels

labels = parse_tracklet_labels(r'C:\Users\n2309064h\Desktop\Multimodal_code\kitti\2011_09_26\2011_09_26_drive_0005_sync\tracklet_labels.xml')
print(labels)
