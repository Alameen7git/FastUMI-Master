import rospy
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped

path = Path()

def callback(msg):
    path.header = msg.header
    path.header.frame_id = "camera_odom_frame"
    pose = PoseStamped()
    pose.header = msg.header
    pose.pose = msg.pose.pose
    path.poses.append(pose)
    pub.publish(path)

rospy.init_node('odom_to_path')
pub = rospy.Publisher('/camera/path', Path, queue_size=10)
rospy.Subscriber('/camera/odom/sample', Odometry, callback)
rospy.spin()
