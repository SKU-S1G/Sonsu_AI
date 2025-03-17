import {
    StyleSheet,
    View,
    Text,
    TouchableOpacity,
    Platform,
  } from "react-native";
  import { useRoute, useNavigation } from "@react-navigation/native";
  import { SafeAreaView } from "react-native-safe-area-context";
  import SpeedBack from "../../components/SpeedBack";
  import { WebView } from "react-native-webview";
  
  export default function Study() {
    const route = useRoute();
    const { topic, lesson } = route.params;
    const navigation = useNavigation();
  
    const serverIP = "http://192.168.10.20:5001";
  
    return (
      <SafeAreaView style={styles.container}>
        <SpeedBack heightMultiplier={1.88} />
  
        <TouchableOpacity
          style={styles.backButton}
          onPress={() => navigation.goBack()}
        >
          <View style={styles.screenContainer}>
            <Text style={styles.title}>
              {"Step "}
              {lesson.id}. {topic}
            </Text>
          </View>
        </TouchableOpacity>
  
        <View style={styles.desContainer}>
          <Text style={{ fontSize: 20, fontWeight: "bold" }}>혼자해보기</Text>
        </View>
  
        <View style={styles.cameraFeedWrapper}>
          {Platform.OS === "web" ? (
            <iframe
              src={`${serverIP}/video_feed`}
              style={styles.cameraFeed}
              width="100%"
              height="100%"
              allowFullScreen
            />
          ) : (
            <WebView
              source={{ uri: `${serverIP}/video_feed` }}
              style={styles.cameraFeed}
              javaScriptEnabled={true}
              domStorageEnabled={true}
              originWhitelist={["*"]}
              allowsFullscreenVideo={true}
              allowsInlineMediaPlayback={true}
              mediaPlaybackRequiresUserAction={false}
              onError={(error) => console.log("WebView error:", error)}
              onHttpError={(syntheticEvent) => {
                const { nativeEvent } = syntheticEvent;
                console.log("HTTP error: ", nativeEvent);
              }}
            />
          )}
        </View>
      </SafeAreaView>
    );
  }
  
  const styles = StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: "#f5f5f5",
      alignItems: "center",
    },
    screenContainer: {
      alignItems: "flex-start",
      width: 350,
    },
    title: {
      fontSize: 22,
      marginTop: 20,
      fontWeight: "bold",
    },
    backButton: {
      padding: 10,
      flexDirection: "row",
    },
    desContainer: {
      marginTop: 30,
      width: 350,
    },
    cameraFeedWrapper: {
      width: 320,
      marginTop: 40,
      aspectRatio: 16 / 9,
    },
    cameraFeed: {
      flex: 1,
      backgroundColor: "transparent",
    },
  });
  